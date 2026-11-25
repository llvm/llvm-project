// Check that the LLVM ABI library passes an atomic kernel pointer the same way
// as Clang's own ABI lowering. Kept apart from abi-lowering.cpp, which also
// runs ClangIR, because ClangIR cannot pass _Atomic values yet.

// RUN: %clang_cc1 -triple spirv64 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-intel -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-intel -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64 -x cuda -fcuda-is-device -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64 -x cuda -fcuda-is-device -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s

#ifdef __CUDA__
#define KERNEL __attribute__((global))
#else
#define KERNEL __attribute__((device_kernel))
#endif

// An atomic pointer of the same size is passed as the pointer itself.
// CHECK: define {{.*}}spir_kernel void @_Z8k_atomicU7_AtomicPi(ptr addrspace(1) %{{[^,)]+}})
KERNEL void k_atomic(_Atomic(int *) p) {}
