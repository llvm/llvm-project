// Check that the LLVM ABI library and ClangIR classify SPIR-V arguments and
// return values the same way as Clang's own ABI lowering.

// The default address space is the private one (PRIV) for plain SPIR-V, and
// the generic one (GEN) for CUDA and the Intel SPIR-V triple.
// RUN: %clang_cc1 -triple spirv64 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,PRIV
// RUN: %clang_cc1 -triple spirv64 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,PRIV
// RUN: %if cir-enabled %{ %clang_cc1 -triple spirv64 -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,PRIV %}
// RUN: %clang_cc1 -triple spirv64-intel -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,GEN
// RUN: %clang_cc1 -triple spirv64-intel -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,GEN
// RUN: %if cir-enabled %{ %clang_cc1 -triple spirv64-intel -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,GEN %}
// RUN: %clang_cc1 -triple spirv64 -x cuda -fcuda-is-device -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DEVICE,GEN
// RUN: %clang_cc1 -triple spirv64 -x cuda -fcuda-is-device -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DEVICE,GEN
// RUN: %if cir-enabled %{ %clang_cc1 -triple spirv64 -x cuda -fcuda-is-device -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DEVICE,GEN %}

// AMDGCN-flavored SPIR-V keeps using Clang's AMDGPU-based lowering.
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x cuda -fcuda-is-device -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefix=AMDGCN

#ifdef __CUDA__
#define DEVICE __attribute__((device))
#define KERNEL __attribute__((global))
#else
#define DEVICE
#define KERNEL __attribute__((device_kernel))
#endif

struct Small {
  int a, b;
};

struct NonTrivial {
  DEVICE NonTrivial(const NonTrivial &);
  int a;
};

struct NoCopy {
  DEVICE NoCopy();
  DEVICE NoCopy(const NoCopy &);
  DEVICE ~NoCopy();
  int a;
};

struct Empty {};
struct TrivialBase {
  int i;
};
struct Derived : TrivialBase {
  int j;
};
typedef int TrivialBase::*mdp_t;
typedef void (TrivialBase::*mfp_t)();

// CHECK: define {{.*}}spir_func noundef signext i8 @_Z3extat(i8 noundef signext %{{[^,)]+}}, i16 noundef zeroext %{{[^,)]+}})
DEVICE signed char ext(signed char c, unsigned short s) { return c + s; }

// Pointers keep their address space outside kernels.
// PRIV: define {{.*}}spir_func void @_Z8take_ptrPi(ptr noundef %{{[^,)]+}})
// GEN:  define {{.*}}spir_func void @_Z8take_ptrPi(ptr addrspace(4) noundef %{{[^,)]+}})
DEVICE void take_ptr(int *p) {}

// CHECK: define {{.*}}spir_func void @_Z9ret_small5Small(ptr dead_on_unwind noalias writable sret(%struct.Small) align 4 %{{[^,)]+}}, ptr noundef byval(%struct.Small) align 4 %{{[^,)]+}})
// AMDGCN: define {{.*}}spir_func [2 x i32] @_Z9ret_small5Small([2 x i32] %s.coerce)
DEVICE Small ret_small(Small s) { return s; }

// Trivially copyable records, with a base or empty, are copied byval and
// returned through a private sret.
// CHECK: define {{.*}}spir_func void @_Z11arg_derived7Derived(ptr noundef byval(%struct.Derived) align 4 %{{[^,)]+}})
DEVICE void arg_derived(Derived d) {}

// CHECK: define {{.*}}spir_func void @_Z11ret_derivedv(ptr dead_on_unwind noalias writable sret(%struct.Derived) align 4 %{{[^,)]+}})
DEVICE Derived ret_derived() { return {}; }

// CHECK: define {{.*}}spir_func void @_Z9arg_empty5Emptyi(ptr noundef byval(%struct.Empty) align 1 %{{[^,)]+}}, i32 noundef %{{[^,)]+}})
DEVICE void arg_empty(Empty e, int i) {}

// CHECK: define {{.*}}spir_func void @_Z9ret_emptyv(ptr dead_on_unwind noalias writable sret(%struct.Empty) align 1 %{{[^,)]+}})
DEVICE Empty ret_empty() { return {}; }

// A data member pointer is an offset, passed and returned as an integer.
// CHECK: define {{.*}}spir_func void @_Z7arg_mdpM11TrivialBasei(i64 %{{[^,)]+}})
DEVICE void arg_mdp(mdp_t p) {}

// CHECK: define {{.*}}spir_func i64 @_Z7ret_mdpM11TrivialBasei(i64 %{{[^,)]+}})
DEVICE mdp_t ret_mdp(mdp_t p) { return p; }

// A member function pointer is a pair, so it is copied byval and returned
// through sret.
// CHECK: define {{.*}}spir_func void @_Z7arg_mfpM11TrivialBaseFvvE(ptr noundef byval({ i64, i64 }) align 8 %{{[^,)]+}})
DEVICE void arg_mfp(mfp_t p) {}

// CHECK: define {{.*}}spir_func void @_Z7ret_mfpM11TrivialBaseFvvE(ptr dead_on_unwind noalias writable sret({ i64, i64 }) align 8 %{{[^,)]+}}, ptr noundef byval({ i64, i64 }) align 8 %{{[^,)]+}})
DEVICE mfp_t ret_mfp(mfp_t p) { return p; }

// A record that cannot be copied bitwise is passed as the address of the
// caller's copy, not byval.
// CHECK: define {{.*}}spir_func void @_Z15take_nontrivial10NonTrivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{[^,)]+}})
DEVICE void take_nontrivial(NonTrivial n) {}

// CHECK-LABEL: define {{.*}}spir_func void @_Z15call_nontrivialR10NonTrivial(
// CHECK:   call spir_func {{.*}}void @_Z15take_nontrivial10NonTrivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{[^,)]+}})
DEVICE void call_nontrivial(NonTrivial &n) { take_nontrivial(n); }

// A record that cannot be copied is constructed in place, so its sret pointer
// is in the default address space.
// CHECK-LABEL: define {{.*}}spir_func void @_Z14ret_nontrivialv(
// PRIV-SAME: ptr dead_on_unwind noalias writable sret(%struct.NoCopy) align 4 %[[RET:[^,)]+]])
// GEN-SAME:  ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NoCopy) align 4 %[[RET:[^,)]+]])
// PRIV:   call spir_func {{.*}}void @_ZN6NoCopyC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %[[RET]])
// GEN:    call spir_func {{.*}}void @_ZN6NoCopyC1Ev(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %[[RET]])
DEVICE NoCopy ret_nontrivial() { return NoCopy(); }

// CHECK-LABEL: define {{.*}}spir_func void @_Z14use_nontrivialv()
// PRIV:   call spir_func {{.*}}void @_Z14ret_nontrivialv(ptr dead_on_unwind writable sret(%struct.NoCopy) align 4 %{{[^,)]+}})
// GEN:    call spir_func {{.*}}void @_Z14ret_nontrivialv(ptr addrspace(4) dead_on_unwind writable sret(%struct.NoCopy) align 4 %{{[^,)]+}})
DEVICE void use_nontrivial() { NoCopy x = ret_nontrivial(); }

// CHECK: define {{.*}}spir_kernel void @_Z7k_small5Small(ptr noundef byval(%struct.Small) align 4 %{{[^,)]+}})
KERNEL void k_small(Small s) {}

// Device compilations copy kernel aggregates byval even when they cannot be
// passed by value otherwise.
// NODEV:  define {{.*}}spir_kernel void @_Z12k_nontrivial10NonTrivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %{{[^,)]+}})
// DEVICE: define {{.*}}spir_kernel void @_Z12k_nontrivial10NonTrivial(ptr noundef byval(%struct.NonTrivial) align 4 %{{[^,)]+}})
KERNEL void k_nontrivial(NonTrivial n) {}

// Default address space kernel pointers are passed as CrossWorkgroup ones; the
// body still sees a default address space pointer.
// CHECK-LABEL: define {{.*}}spir_kernel void @_Z5k_ptrPi(ptr addrspace(1) noundef %{{[^,)]+}})
// PRIV:   call spir_func {{.*}}void @_Z8take_ptrPi(ptr noundef %{{[^,)]+}})
// GEN:    call spir_func {{.*}}void @_Z8take_ptrPi(ptr addrspace(4) noundef %{{[^,)]+}})
KERNEL void k_ptr(int *p) { take_ptr(p); }

// So are references.
// PRIV: define {{.*}}spir_kernel void @_Z5k_refRi(ptr addrspace(1) noundef nonnull align 4 dereferenceable(4) %{{[^,)]+}})
// GEN:  define {{.*}}spir_kernel void @_Z5k_refRi(ptr addrspace(1) noundef align 4 dereferenceable(4) %{{[^,)]+}})
KERNEL void k_ref(int &r) {}

// std::nullptr_t is an address space 0 pointer, so it is only coerced where
// that is the default address space.
// PRIV: define {{.*}}spir_kernel void @_Z9k_nullptrDn(ptr addrspace(1) %{{[^,)]+}})
// GEN:  define {{.*}}spir_kernel void @_Z9k_nullptrDn(ptr %{{[^,)]+}})
KERNEL void k_nullptr(decltype(nullptr) n) {}
