// RUN: echo -n "GPU binary would be here." > %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=11.0 -fcuda-include-gpubinary %t -o - \
// RUN: | FileCheck %s --check-prefixes CUDA
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x hip \
// RUN:     -fcuda-include-gpubinary %t -o - \
// RUN: | FileCheck %s --check-prefixes HIP

#include "Inputs/cuda.h"

template <typename T>
struct S { T t; };

template <typename T>
    static __global__ void Kernel(S<T>) {}

// For some reason it takes three or more instantiations of Kernel to trigger a
// crash during CUDA compilation.
auto x = &Kernel<double>;
auto y = &Kernel<float>;
auto z = &Kernel<int>;

// This triggers HIP-specific code path.
void func (){
  Kernel<short><<<1,1>>>({1});
}

// CUDA-LABEL: @__cuda_register_globals(
// CUDA:  call i32 @__cudaRegisterFunction(ptr %0, ptr @_ZL21__device_stub__KernelIdEv1SIT_E
// CUDA:  call i32 @__cudaRegisterFunction(ptr %0, ptr @_ZL21__device_stub__KernelIfEv1SIT_E
// CUDA:  call i32 @__cudaRegisterFunction(ptr %0, ptr @_ZL21__device_stub__KernelIiEv1SIT_E
// CUDA:  ret void

// HIP-LABEL: @__hip_register_globals(
// HIP:   call i32 @__hipRegisterFunction(ptr %0, ptr @_ZL6KernelIdEv1SIT_E
// HIP:   call i32 @__hipRegisterFunction(ptr %0, ptr @_ZL6KernelIfEv1SIT_E
// HIP:   call i32 @__hipRegisterFunction(ptr %0, ptr @_ZL6KernelIiEv1SIT_E
// HIP:   ret void
