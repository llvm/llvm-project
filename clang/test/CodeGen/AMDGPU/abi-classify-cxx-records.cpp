// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fexperimental-abi-lowering -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fexperimental-abi-lowering -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %if cir-enabled %{ %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,GCN %}

// Checks how AMDGPU functions pass and return C++ records, where triviality
// and layout change the result.

struct Empty {};

struct NonTrivialDtor {
  int i;
  ~NonTrivialDtor();
};

struct NonCopyable {
  int i;
  NonCopyable();
  NonCopyable(const NonCopyable &) = delete;
};

struct TrivialBase { int i; };
struct DerivedSingle : TrivialBase {};
struct DerivedPair : TrivialBase { int j; };
struct DerivedWithEmptyBase : Empty { float f; };
struct MemberPtrHolder { int TrivialBase::*p; };
typedef void (TrivialBase::*mfp_t)();

// A record that cannot be passed in registers is built directly in a generic
// sret.
NonTrivialDtor ret_non_trivial_dtor() { return {1}; }

// GCN: define {{.*}}void @_Z20ret_non_trivial_dtorv(ptr dead_on_unwind noalias writable sret(%struct.NonTrivialDtor) align 4 %[[RET:.*]])
// SPIRV: define {{.*}}void @_Z20ret_non_trivial_dtorv(ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NonTrivialDtor) align 4 %{{.*}})
// GCN:   %[[I:.*]] = getelementptr inbounds nuw %struct.NonTrivialDtor, ptr %[[RET]], i32 0, i32 0
// GCN:   store i32 1, ptr %[[I]], align 4

// The caller's local is built in place by the callee.
void use_non_trivial_dtor() { NonTrivialDtor x = ret_non_trivial_dtor(); }

// CHECK: define {{.*}}void @_Z20use_non_trivial_dtorv()
// GCN:   call void @_Z20ret_non_trivial_dtorv(ptr dead_on_unwind writable sret(%struct.NonTrivialDtor) align 4 %{{.*}})

NonCopyable ret_non_copyable() { return NonCopyable(); }

// GCN: define {{.*}}void @_Z16ret_non_copyablev(ptr dead_on_unwind noalias writable sret(%struct.NonCopyable) align 4 %{{.*}})
// SPIRV: define {{.*}}void @_Z16ret_non_copyablev(ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NonCopyable) align 4 %{{.*}})

// A base subobject holding the only field still counts as a single element.
DerivedSingle ret_derived_single() { return {}; }

// CHECK: define {{.*}}i32 @_Z18ret_derived_singlev()

void arg_derived_single(DerivedSingle d) {}

// CHECK: define {{.*}}void @_Z18arg_derived_single13DerivedSingle(i32 %{{.*}})

// An empty base does not count as an element.
DerivedWithEmptyBase ret_derived_empty_base() { return {}; }

// CHECK: define {{.*}}float @_Z22ret_derived_empty_basev()

void arg_derived_empty_base(DerivedWithEmptyBase d) {}

// CHECK: define {{.*}}void @_Z22arg_derived_empty_base20DerivedWithEmptyBase(float %{{.*}})

// A base plus a field is packed into a pair of i32 registers.
DerivedPair ret_derived_pair() { return {}; }

// CHECK: define {{.*}}[2 x i32] @_Z16ret_derived_pairv()
// GCN:   %[[RET:.*]] = load [2 x i32], ptr addrspace(5) %{{.*}}, align 4
// GCN:   ret [2 x i32] %[[RET]]

void arg_derived_pair(DerivedPair d) {}

// CHECK: define {{.*}}void @_Z16arg_derived_pair11DerivedPair([2 x i32] %[[D:[^)]+]])
// GCN:   store [2 x i32] %[[D]], ptr addrspace(5) %{{.*}}, align 4

// A data member pointer is passed as its i64 offset.
MemberPtrHolder ret_member_ptr() { return {nullptr}; }

// CHECK: define {{.*}}i64 @_Z14ret_member_ptrv()

void arg_member_ptr(MemberPtrHolder h) {}

// CHECK: define {{.*}}void @_Z14arg_member_ptr15MemberPtrHolder(i64 %{{.*}})

// A member function pointer is a pair of i64 that passes one field per
// register.
void arg_mfp(mfp_t f) {}

// CHECK: define {{.*}}void @_Z7arg_mfpM11TrivialBaseFvvE(i64 %[[PTR:[^,]+]], i64 %[[ADJ:[^)]+]])
// GCN:   store i64 %[[PTR]], ptr addrspace(5) %{{.*}}, align 8
// GCN:   store i64 %[[ADJ]], ptr addrspace(5) %{{.*}}, align 8

mfp_t ret_mfp() { return nullptr; }

// CHECK: define {{.*}}{ i64, i64 } @_Z7ret_mfpv()

// An empty C++ record is one byte in size but is still dropped.
void arg_empty(Empty e) {}

// CHECK: define {{.*}}void @_Z9arg_empty5Empty()

// A non-trivial record argument is passed through a pointer in addrspace(5).
void arg_non_trivial_dtor(NonTrivialDtor n) {}

// GCN: define {{.*}}void @_Z20arg_non_trivial_dtor14NonTrivialDtor(ptr addrspace(5) nofreeobj noundef align 4 dereferenceable(4) %{{.*}})
// SPIRV: define {{.*}}void @_Z20arg_non_trivial_dtor14NonTrivialDtor(ptr nofreeobj noundef align 4 dereferenceable(4) %{{.*}})

// The caller passes its copied temporary, not a further copy.
void call_non_trivial_dtor(NonTrivialDtor &n) { arg_non_trivial_dtor(n); }

// CHECK: define {{.*}}void @_Z21call_non_trivial_dtorR14NonTrivialDtor(
// GCN:   %[[TMP:.*]] = alloca %struct.NonTrivialDtor, align 4, addrspace(5)
// GCN:   %[[TMP_VIEW:.*]] = addrspacecast ptr addrspace(5) %[[TMP]] to ptr
// GCN:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP_VIEW]], ptr align 4 %{{.*}}, i64 4, i1 false)
// GCN:   call void @_Z20arg_non_trivial_dtor14NonTrivialDtor(ptr addrspace(5) nofreeobj noundef align 4 dereferenceable(4) %{{.*}})
// GCN:   call void @_ZN14NonTrivialDtorD1Ev(ptr {{.*}}%[[TMP_VIEW]])
