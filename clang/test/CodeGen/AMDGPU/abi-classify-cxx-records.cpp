// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV

// Check that the ABI library classifies C++ records the same way Clang does.

struct NonTrivialDtor { int i; ~NonTrivialDtor(); };
struct NonCopyable { int i; NonCopyable(const NonCopyable &) = delete; NonCopyable(); };
struct TrivialBase { int i; };
struct DerivedSingle : TrivialBase {};
struct EmptyBase {};
struct DerivedWithEmptyBase : EmptyBase { float f; };
struct MemberPtrHolder { int TrivialBase::*p; };

// Constructed in place, so the sret pointer uses the generic address space.
NonTrivialDtor ret_non_trivial_dtor() { return NonTrivialDtor(); }
// GCN: define{{.*}} void @_Z20ret_non_trivial_dtorv(ptr dead_on_unwind noalias writable sret(%struct.NonTrivialDtor) align 4 %{{.*}})
// SPIRV: define{{.*}} void @_Z20ret_non_trivial_dtorv(ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NonTrivialDtor) align 4 %{{.*}})

NonCopyable ret_non_copyable() { return NonCopyable(); }
// GCN: define{{.*}} void @_Z16ret_non_copyablev(ptr dead_on_unwind noalias writable sret(%struct.NonCopyable) align 4 %{{.*}})
// SPIRV: define{{.*}} void @_Z16ret_non_copyablev(ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NonCopyable) align 4 %{{.*}})

DerivedSingle ret_derived_single() { return DerivedSingle(); }
// CHECK: define{{.*}} i32 @_Z18ret_derived_singlev()

DerivedWithEmptyBase ret_derived_empty_base() { return DerivedWithEmptyBase(); }
// CHECK: define{{.*}} float @_Z22ret_derived_empty_basev()

// A data member pointer is a scalar, so the wrapper is a single-element struct.
MemberPtrHolder ret_member_ptr() { return MemberPtrHolder(); }
// CHECK: define{{.*}} i64 @_Z14ret_member_ptrv()

// By contrast an argument passed by address uses the alloca address space.
void arg_non_trivial_dtor(NonTrivialDtor s) {}
// GCN: define{{.*}} void @_Z20arg_non_trivial_dtor14NonTrivialDtor(ptr addrspace(5) nofreeobj noundef align 4 dereferenceable(4) %{{.*}})
// SPIRV: define{{.*}} void @_Z20arg_non_trivial_dtor14NonTrivialDtor(ptr nofreeobj noundef align 4 dereferenceable(4) %{{.*}})

void arg_derived_single(DerivedSingle s) {}
// CHECK: define{{.*}} void @_Z18arg_derived_single13DerivedSingle(i32 %{{.*}})

void arg_derived_empty_base(DerivedWithEmptyBase s) {}
// CHECK: define{{.*}} void @_Z22arg_derived_empty_base20DerivedWithEmptyBase(float %{{.*}})

void arg_member_ptr(MemberPtrHolder s) {}
// CHECK: define{{.*}} void @_Z14arg_member_ptr15MemberPtrHolder(i64 %{{.*}})
