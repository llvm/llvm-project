#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

__global__ void fn() {
  int i = 0;
  __shared__ int j;
  j = i;
}

// CIR-DEVICE: cir.global "private" internal dso_local @_ZZ2fnvE1j : !s32i
// CIR-DEVICE: cir.func {{.*}}@_Z2fnv() {{.*}} {
// CIR-DEVICE:   %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR-DEVICE:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE:   cir.store {{.*}}%[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:   %[[J:.*]] = cir.get_global @_ZZ2fnvE1j : !cir.ptr<!s32i>
// CIR-DEVICE:   %[[VAL:.*]] = cir.load {{.*}}%[[I]] : !cir.ptr<!s32i>, !s32i
// CIR-DEVICE:   cir.store {{.*}}%[[VAL]], %[[J]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:   cir.return

// CIR-HOST: cir.func private dso_local @__cudaPopCallConfiguration
// CIR-HOST: cir.func private dso_local @cudaLaunchKernel
// CIR-HOST: cir.func {{.*}}@_Z17__device_stub__fnv()

// LLVM-DEVICE: @_ZZ2fnvE1j = internal global i32 undef, align 4
// LLVM-DEVICE: define dso_local void @_Z2fnv()
// LLVM-DEVICE:   %[[ALLOCA:.*]] = alloca i32, i64 1, align 4
// LLVM-DEVICE:   store i32 0, ptr %[[ALLOCA]], align 4
// LLVM-DEVICE:   %[[VAL:.*]] = load i32, ptr %[[ALLOCA]], align 4
// LLVM-DEVICE:   store i32 %[[VAL]], ptr @_ZZ2fnvE1j, align 4
// LLVM-DEVICE:   ret void

// LLVM-HOST: %struct.dim3 = type { i32, i32, i32 }
// LLVM-HOST: declare {{.*}}i32 @__cudaPopCallConfiguration(ptr, ptr, ptr, ptr)
// LLVM-HOST: declare {{.*}}i32 @cudaLaunchKernel(ptr, %struct.dim3, %struct.dim3, ptr, i64, ptr)
// LLVM-HOST: define dso_local void @_Z17__device_stub__fnv()

// OGCG-HOST: define dso_local void @_Z17__device_stub__fnv()
// OGCG-HOST: entry:
// OGCG-HOST:   call i32 @__cudaPopCallConfiguration
// OGCG-HOST:   call {{.*}}i32 @cudaLaunchKernel

// OGCG-DEVICE: @_ZZ2fnvE1j = internal addrspace(3) global i32 undef, align 4
// OGCG-DEVICE: define dso_local ptx_kernel void @_Z2fnv()
// OGCG-DEVICE: entry:
// OGCG-DEVICE:   %[[I:.*]] = alloca i32, align 4
// OGCG-DEVICE:   store i32 0, ptr %[[I]], align 4
// OGCG-DEVICE:   %[[VAL:.*]] = load i32, ptr %[[I]], align 4
// OGCG-DEVICE:   store i32 %[[VAL]], ptr addrspacecast (ptr addrspace(3) @_ZZ2fnvE1j to ptr), align 4
// OGCG-DEVICE:   ret void
