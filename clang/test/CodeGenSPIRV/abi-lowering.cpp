// Check that the LLVM ABI library classifies SPIR-V arguments and return
// values the same way as Clang's own ABI lowering.

// The default address space is the private one (PRIV) for plain SPIR-V, and
// the generic one (GEN) for CUDA and the Intel SPIR-V triple.
// RUN: %clang_cc1 -triple spirv64 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,PRIV
// RUN: %clang_cc1 -triple spirv64 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,PRIV
// RUN: %clang_cc1 -triple spirv64-intel -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,GEN
// RUN: %clang_cc1 -triple spirv64-intel -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NODEV,GEN
// RUN: %clang_cc1 -triple spirv64 -x cuda -fcuda-is-device -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DEVICE,GEN
// RUN: %clang_cc1 -triple spirv64 -x cuda -fcuda-is-device -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DEVICE,GEN

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

// CHECK: define {{.*}}spir_func noundef signext i8 @_Z3extat(i8 noundef signext %c, i16 noundef zeroext %s)
DEVICE signed char ext(signed char c, unsigned short s) { return c + s; }

// CHECK: define {{.*}}spir_func void @_Z9ret_small5Small(ptr dead_on_unwind noalias writable sret(%struct.Small) align 4 %agg.result, ptr noundef byval(%struct.Small) align 4 %s)
// AMDGCN: define {{.*}}spir_func [2 x i32] @_Z9ret_small5Small([2 x i32] %s.coerce)
DEVICE Small ret_small(Small s) { return s; }

// CHECK: define {{.*}}spir_func void @_Z15take_nontrivial10NonTrivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %n)
DEVICE void take_nontrivial(NonTrivial n) {}

// A record that cannot be copied is constructed in place, so its sret pointer
// is in the default address space.
// PRIV: define {{.*}}spir_func void @_Z14ret_nontrivialv(ptr dead_on_unwind noalias writable sret(%struct.NoCopy) align 4 %agg.result)
// GEN:  define {{.*}}spir_func void @_Z14ret_nontrivialv(ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NoCopy) align 4 %agg.result)
DEVICE NoCopy ret_nontrivial() { return NoCopy(); }

// CHECK: define {{.*}}spir_kernel void @_Z7k_small5Small(ptr noundef byval(%struct.Small) align 4 %s)
KERNEL void k_small(Small s) {}

// Device compilations copy kernel aggregates byval even when they cannot be
// passed by value otherwise.
// NODEV:  define {{.*}}spir_kernel void @_Z12k_nontrivial10NonTrivial(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %n)
// DEVICE: define {{.*}}spir_kernel void @_Z12k_nontrivial10NonTrivial(ptr noundef byval(%struct.NonTrivial) align 4 %n)
KERNEL void k_nontrivial(NonTrivial n) {}

// Default address space kernel pointers are passed as CrossWorkgroup ones.
// CHECK: define {{.*}}spir_kernel void @_Z5k_ptrPi(ptr addrspace(1) noundef %p.coerce)
KERNEL void k_ptr(int *p) {}

// CHECK: define {{.*}}spir_kernel void @_Z8k_atomicU7_AtomicPi(ptr addrspace(1) %p.coerce)
KERNEL void k_atomic(_Atomic(int *) p) {}

// std::nullptr_t is an address space 0 pointer, so it is only coerced where
// that is the default address space.
// PRIV: define {{.*}}spir_kernel void @_Z9k_nullptrDn(ptr addrspace(1) %n.coerce)
// GEN:  define {{.*}}spir_kernel void @_Z9k_nullptrDn(ptr %n)
KERNEL void k_nullptr(decltype(nullptr) n) {}
