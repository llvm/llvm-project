#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fclangir -fcuda-is-device -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fclangir -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s

// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s

int global_host_var;
__device__ int global_device_var;

template <class F>
__global__ void kern(F f) { f(); }

__device__ void dev_capture_dev_ref_by_copy(int *out) {
  int &ref = global_device_var;
  [=]() { *out = ref; }();
}

// CIR-LABEL: cir.func {{.*}} @_ZZ27dev_capture_dev_ref_by_copyPiENKUlvE_clEv(
// CIR:         %[[GV:.*]] = cir.const #cir.global_view<@global_device_var> : !cir.ptr<!s32i>
// CIR-NEXT:    %[[VAL:.*]] = cir.load align(4) %[[GV]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store align(4) %[[VAL]], %{{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define internal void @_ZZ27dev_capture_dev_ref_by_copyPiENKUlvE_clEv(
// LLVM:         %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr), align 4
// LLVM:         store i32 %[[VAL]], ptr %{{.*}}, align 4

// The host variable is not accessible on the device, so the value must be
// loaded from the lambda capture rather than from @global_host_var.
void dev_capture_host_ref_by_copy(int *out) {
  int &ref = global_host_var;
  kern<<<1, 1>>>([=] __device__() { *out = ref; });
}

// CIR-LABEL: cir.func {{.*}} @_ZZ28dev_capture_host_ref_by_copyPiENKUlvE_clEv(
// CIR:         %[[THIS:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<![[ANON:.*]]>>, !cir.ptr<![[ANON]]>
// CIR-NEXT:    %[[CAP:.*]] = cir.get_member %[[THIS]][1] {name = ""} : !cir.ptr<![[ANON]]> -> !cir.ptr<!s32i>
// CIR-NEXT:    %[[VAL:.*]] = cir.load align(8) %[[CAP]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store align(4) %[[VAL]], %{{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define internal void @_ZZ28dev_capture_host_ref_by_copyPiENKUlvE_clEv(
// LLVM:         %[[THIS:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-NEXT:    %[[CAP:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %[[THIS]], i32 0, i32 1
// LLVM-NEXT:    %[[VAL:.*]] = load i32, ptr %[[CAP]], align 8
// LLVM:         store i32 %[[VAL]], ptr %{{.*}}, align 4
