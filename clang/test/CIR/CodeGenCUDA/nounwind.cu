#include "Inputs/cuda.h"

// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=LLVM

// Device code cannot unwind even when exceptions are enabled, so calls inside
// an EH cleanup scope must stay plain calls.
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fcxx-exceptions -fexceptions -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=LLVM

struct D {
  __device__ ~D();
};

extern "C" {
__device__ int ext(int);

__device__ int caller(int x) {
  D d;
  return ext(x);
}

__global__ void kernel(int *p) { *p = caller(*p); }
}

// CIR: cir.func{{.*}}@caller(
// CIR-SAME: nothrow, nounwind
// CIR-NOT: cir.try_call
// CIR: cir.call @ext(%{{.*}}) nothrow nounwind
// CIR-NOT: cir.try_call
// CIR: cir.call @_ZN1DD1Ev(%{{.*}}) nothrow nounwind
// CIR-NOT: cir.try_call
// CIR: cir.func private @ext(
// CIR-SAME: nothrow, nounwind
// CIR: cir.func{{.*}}@kernel(
// CIR-SAME: nothrow, nounwind
// CIR: cir.call @caller(%{{.*}}) nothrow nounwind

// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: define {{.*}}@caller({{.*}}){{.*}} #{{[0-9]+}} {
// LLVM-NOT: invoke
// LLVM: call {{.*}}@ext({{.*}}) #[[CALL_ATTR:[0-9]+]]
// LLVM-NOT: invoke
// LLVM: call void @_ZN1DD1Ev({{.*}}) #[[CALL_ATTR]]
// LLVM-NOT: landingpad
// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: declare {{.*}}@ext(
// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: declare {{.*}}@_ZN1DD1Ev(
// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: define {{.*}}ptx_kernel void @kernel({{.*}}){{.*}} #{{[0-9]+}} {
// LLVM: call {{.*}}@caller({{.*}}) #[[CALL_ATTR]]
// LLVM: attributes #[[CALL_ATTR]] = {{{.*}}nounwind
