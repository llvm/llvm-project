// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies C++ records on ordinary AMDGPU
// functions, where C++ rules on triviality and layout change the result.

// TODO(cir): Add the C++ record cases still NYI in CallConvLowering:
// - Non-trivial records as arguments. Classic passes them indirectly as
//   ptr addrspace(5), but the bridge drops the indirect address space.
// - Member function pointers, lowered to a 128-bit pair that the bridge
//   cannot coerce yet.
// - Records of 33 to 64 bits, such as a base plus a field, which classic
//   coerces to [2 x i32].

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
struct MemberPtrHolder { int TrivialBase::*p; };

// A record that cannot be passed in registers returns through a generic sret.
NonTrivialDtor ret_non_trivial_dtor() { return {1}; }

// CIR: cir.func {{.*}}@_Z20ret_non_trivial_dtorv(%arg0: !cir.ptr<!rec_NonTrivialDtor> {{.*}}llvm.sret = !rec_NonTrivialDtor
// LLVM: define {{.*}}void @_Z20ret_non_trivial_dtorv(ptr dead_on_unwind noalias writable sret(%struct.NonTrivialDtor) align 4 %{{.*}})

NonCopyable ret_non_copyable() { return NonCopyable(); }

// CIR: cir.func {{.*}}@_Z16ret_non_copyablev(%arg0: !cir.ptr<!rec_NonCopyable> {{.*}}llvm.sret = !rec_NonCopyable
// LLVM: define {{.*}}void @_Z16ret_non_copyablev(ptr dead_on_unwind noalias writable sret(%struct.NonCopyable) align 4 %{{.*}})

// A base subobject holding the only field still counts as a single element.
DerivedSingle ret_derived_single() { return {}; }

// CIR: cir.func {{.*}}@_Z18ret_derived_singlev() -> !s32i
// LLVM: define {{.*}}i32 @_Z18ret_derived_singlev()

void arg_derived_single(DerivedSingle d) {}

// CIR: cir.func {{.*}}@_Z18arg_derived_single13DerivedSingle(%arg0: !s32i{{.*}})
// LLVM: define {{.*}}void @_Z18arg_derived_single13DerivedSingle(i32 %{{.*}})

// A data member pointer is lowered to an i64 offset before classification.
MemberPtrHolder ret_member_ptr() { return {nullptr}; }

// CIR: cir.func {{.*}}@_Z14ret_member_ptrv() -> !s64i
// LLVM: define {{.*}}i64 @_Z14ret_member_ptrv()

void arg_member_ptr(MemberPtrHolder h) {}

// CIR: cir.func {{.*}}@_Z14arg_member_ptr15MemberPtrHolder(%arg0: !s64i{{.*}})
// LLVM: define {{.*}}void @_Z14arg_member_ptr15MemberPtrHolder(i64 %{{.*}})

// An empty C++ record is one byte in size but is still dropped.
void arg_empty(Empty e) {}

// CIR: cir.func {{.*}}@_Z9arg_empty5Empty()
// LLVM: define {{.*}}void @_Z9arg_empty5Empty()
