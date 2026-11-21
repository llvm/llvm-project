// Checks how SPIR-V functions pass and return C++ records, where triviality
// changes the result. Without a device language the default address space is
// the private one and kernels don't copy non-trivial records byval.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s %}

struct Empty {};
struct Small { int a, b; };
struct NonTrivialDtor { int i; ~NonTrivialDtor(); };
struct NonTrivialCopy { int i; NonTrivialCopy(const NonTrivialCopy &); };
struct TrivialBase { int i; };
struct Derived : TrivialBase { int j; };
typedef int TrivialBase::*mdp_t;
typedef void (TrivialBase::*mfp_t)();

// A record that cannot be passed in registers is built directly in the sret.
// CHECK-LABEL: define spir_func void @_Z20ret_non_trivial_dtorv(
// CHECK-SAME: ptr dead_on_unwind noalias writable sret(%struct.NonTrivialDtor) align 4 %[[RET:[^,)]+]])
// CHECK:   %[[I:[^,)]+]] = getelementptr {{.*}}%struct.NonTrivialDtor, ptr %[[RET]], i32 0, i32 0
// CHECK:   store i32 1, ptr %[[I]], align 4
NonTrivialDtor ret_non_trivial_dtor() { return {1}; }

// CHECK-LABEL: define spir_func void @_Z20use_non_trivial_dtorv()
// CHECK:   call spir_func void @_Z20ret_non_trivial_dtorv(ptr dead_on_unwind writable sret(%struct.NonTrivialDtor) align 4 %{{[^,)]+}})
void use_non_trivial_dtor() { NonTrivialDtor x = ret_non_trivial_dtor(); }

// A record that cannot be copied bitwise is passed as the address of the
// caller's copy, not byval.
// CHECK: define spir_func noundef i32 @_Z20arg_non_trivial_copy14NonTrivialCopy(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{[^,)]+}})
int arg_non_trivial_copy(NonTrivialCopy n) { return n.i; }

// CHECK-LABEL: define spir_func noundef i32 @_Z21call_non_trivial_copyR14NonTrivialCopy(
// CHECK:   call spir_func void @_ZN14NonTrivialCopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP:[^,)]+]], ptr
// CHECK:   call spir_func noundef i32 @_Z20arg_non_trivial_copy14NonTrivialCopy(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[TMP]])
int call_non_trivial_copy(NonTrivialCopy &n) { return arg_non_trivial_copy(n); }

// Trivially copyable records, with a base or empty, are copied byval and
// returned through sret.
// CHECK: define spir_func void @_Z11arg_derived7Derived(ptr noundef byval(%struct.Derived) align 4 %{{[^,)]+}})
void arg_derived(Derived d) {}

// CHECK: define spir_func void @_Z11ret_derivedv(ptr dead_on_unwind noalias writable sret(%struct.Derived) align 4 %{{[^,)]+}})
Derived ret_derived() { return {}; }

// CHECK: define spir_func void @_Z9arg_empty5Emptyi(ptr noundef byval(%struct.Empty) align 1 %{{[^,)]+}}, i32 noundef %{{[^,)]+}})
void arg_empty(Empty e, int i) {}

// CHECK: define spir_func void @_Z9ret_emptyv(ptr dead_on_unwind noalias writable sret(%struct.Empty) align 1 %{{[^,)]+}})
Empty ret_empty() { return {}; }

// A data member pointer is an offset, passed and returned as an integer.
// CHECK: define spir_func void @_Z7arg_mdpM11TrivialBasei(i64 %{{[^,)]+}})
void arg_mdp(mdp_t p) {}

// CHECK: define spir_func i64 @_Z7ret_mdpM11TrivialBasei(i64 %{{[^,)]+}})
mdp_t ret_mdp(mdp_t p) { return p; }

// A member function pointer is a pair, so it is copied byval and returned
// through sret.
// CHECK: define spir_func void @_Z7arg_mfpM11TrivialBaseFvvE(ptr noundef byval({ i64, i64 }) align 8 %{{[^,)]+}})
void arg_mfp(mfp_t p) {}

// CHECK: define spir_func void @_Z7ret_mfpM11TrivialBaseFvvE(ptr dead_on_unwind noalias writable sret({ i64, i64 }) align 8 %{{[^,)]+}}, ptr noundef byval({ i64, i64 }) align 8 %{{[^,)]+}})
mfp_t ret_mfp(mfp_t p) { return p; }

// Kernels copy trivially copyable records byval, but pass a non-trivial one
// like other functions: device languages are what make kernels copy it.
// CHECK: define spir_kernel void @_Z7k_small5Small(ptr noundef byval(%struct.Small) align 4 %{{[^,)]+}})
__attribute__((device_kernel)) void k_small(Small s) {}

// CHECK: define spir_kernel void @_Z18k_non_trivial_copy14NonTrivialCopy(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{[^,)]+}})
__attribute__((device_kernel)) void k_non_trivial_copy(NonTrivialCopy n) {}

// Kernel pointers and references in the default (private) address space are
// passed as CrossWorkgroup ones. std::nullptr_t is such a pointer too.
// CHECK: define spir_kernel void @_Z5k_refRi(ptr addrspace(1) noundef nonnull align 4 dereferenceable(4) %{{[^,)]+}})
__attribute__((device_kernel)) void k_ref(int &r) {}

// CHECK: define spir_kernel void @_Z9k_nullptrDn(ptr addrspace(1) %{{[^,)]+}})
__attribute__((device_kernel)) void k_nullptr(decltype(nullptr) n) {}
