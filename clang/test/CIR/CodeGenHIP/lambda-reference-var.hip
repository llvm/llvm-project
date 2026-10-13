// Based on clang/test/CodeGenCUDA/lambda-reference-var.cu.

// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -x hip -std=c++11 -triple x86_64-linux-gnu \
// RUN:   -fhip-new-launch-api -fclangir -emit-cir -I%S/../CodeGenCUDA/Inputs \
// RUN:   %s -o %t-host.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t-host.cir %s
// RUN: %clang_cc1 -x hip -std=c++11 -triple x86_64-linux-gnu \
// RUN:   -fhip-new-launch-api -fclangir -emit-llvm -I%S/../CodeGenCUDA/Inputs \
// RUN:   %s -o %t-host-cir.ll
// RUN: FileCheck --check-prefix=HOST --input-file=%t-host-cir.ll %s
// RUN: %clang_cc1 -x hip -std=c++11 -triple x86_64-linux-gnu \
// RUN:   -fhip-new-launch-api -emit-llvm -I%S/../CodeGenCUDA/Inputs \
// RUN:   %s -o %t-host.ll
// RUN: FileCheck --check-prefix=HOST --input-file=%t-host.ll %s

// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -fclangir -emit-cir -I%S/../CodeGenCUDA/Inputs %s -o %t-dev.cir
// RUN: FileCheck --check-prefix=CIR-DEV --input-file=%t-dev.cir %s
// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -fclangir -emit-llvm -I%S/../CodeGenCUDA/Inputs %s -o %t-dev-cir.ll
// RUN: FileCheck --check-prefix=DEV --input-file=%t-dev-cir.ll %s
// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -emit-llvm -I%S/../CodeGenCUDA/Inputs %s -o %t-dev.ll
// RUN: FileCheck --check-prefix=DEV --input-file=%t-dev.ll %s

#include "cuda.h"

int global_host_var;
__device__ int global_device_var;

template<class F>
__global__ void kern(F f) { f(); }

// CIR-DEV-LABEL: cir.func {{.*}} @_ZZ27dev_capture_dev_ref_by_copyPiENKUlvE_clEv(
// CIR-DEV:         %[[G:.*]] = cir.const #cir.global_view<@global_device_var> : !cir.ptr<!s32i>
// CIR-DEV:         %[[VAL:.*]] = cir.load {{.*}}%[[G]] : !cir.ptr<!s32i>, !s32i
// CIR-DEV:         cir.store {{.*}}%[[VAL]]
// DEV-LABEL: @_ZZ27dev_capture_dev_ref_by_copyPiENKUlvE_clEv(
// DEV: %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: store i32 %[[VAL]]
__device__ void dev_capture_dev_ref_by_copy(int *out) {
  int &ref = global_device_var;
  [=](){ *out = ref;}();
}

// DEV-LABEL: @_ZZ28dev_capture_dev_rval_by_copyPiENKUlvE_clEv(
// DEV: store i32 3
__device__ void dev_capture_dev_rval_by_copy(int *out) {
  constexpr int a = 1;
  constexpr int b = 2;
  constexpr int c = a + b;
  [=](){ *out = c;}();
}

// DEV-LABEL: @_ZZ26dev_capture_dev_ref_by_refPiENKUlvE_clEv(
// DEV: %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// DEV: store i32 %[[VAL2]], ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: store i32 %[[VAL]]
__device__ void dev_capture_dev_ref_by_ref(int *out) {
  int &ref = global_device_var;
  [&](){ ref++; *out = ref;}();
}

// DEV-LABEL: define{{.*}} void @_Z7dev_refPi(
// DEV: %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// DEV: store i32 %[[VAL2]], ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: store i32 %[[VAL]]
__device__ void dev_ref(int *out) {
  int &ref = global_device_var;
  ref++;
  *out = ref;
}

// DEV-LABEL: @_ZZ14dev_lambda_refPiENKUlvE_clEv(
// DEV: %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// DEV: store i32 %[[VAL2]], ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: %[[VAL:.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @global_device_var to ptr)
// DEV: store i32 %[[VAL]]
__device__ void dev_lambda_ref(int *out) {
  [=](){
    int &ref = global_device_var;
    ref++;
    *out = ref;
  }();
}

// CIR-HOST-LABEL: cir.func {{.*}} @_ZZ29host_capture_host_ref_by_copyPiENKUlvE_clEv(
// CIR-HOST:         %[[G:.*]] = cir.const #cir.global_view<@global_host_var> : !cir.ptr<!s32i>
// CIR-HOST:         %[[VAL:.*]] = cir.load {{.*}}%[[G]] : !cir.ptr<!s32i>, !s32i
// CIR-HOST:         cir.store {{.*}}%[[VAL]]
// HOST-LABEL: @_ZZ29host_capture_host_ref_by_copyPiENKUlvE_clEv(
// HOST: %[[VAL:.*]] = load i32, ptr @global_host_var
// HOST: store i32 %[[VAL]]
void host_capture_host_ref_by_copy(int *out) {
  int &ref = global_host_var;
  [=](){ *out = ref;}();
}

// HOST-LABEL: @_ZZ28host_capture_host_ref_by_refPiENKUlvE_clEv(
// HOST: %[[CAP:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %{{.*}}, i32 0, i32 0
// HOST: %[[REF:.*]] = load ptr, ptr %[[CAP]]
// HOST: %[[VAL:.*]] = load i32, ptr %[[REF]]
// HOST: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// HOST: store i32 %[[VAL2]], ptr %[[REF]]
// HOST: %[[VAL:.*]] = load i32, ptr @global_host_var
// HOST: store i32 %[[VAL]]
void host_capture_host_ref_by_ref(int *out) {
  int &ref = global_host_var;
  [&](){ ref++; *out = ref;}();
}

// HOST-LABEL: define{{.*}} void @_Z8host_refPi(
// HOST: %[[VAL:.*]] = load i32, ptr @global_host_var
// HOST: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// HOST: store i32 %[[VAL2]], ptr @global_host_var
// HOST: %[[VAL:.*]] = load i32, ptr @global_host_var
// HOST: store i32 %[[VAL]]
void host_ref(int *out) {
  int &ref = global_host_var;
  ref++;
  *out = ref;
}

// HOST-LABEL: @_ZZ15host_lambda_refPiENKUlvE_clEv(
// HOST: %[[VAL:.*]] = load i32, ptr @global_host_var
// HOST: %[[VAL2:.*]] = add nsw i32 %[[VAL]], 1
// HOST: store i32 %[[VAL2]], ptr @global_host_var
// HOST: %[[VAL:.*]] = load i32, ptr @global_host_var
// HOST: store i32 %[[VAL]]
void host_lambda_ref(int *out) {
  [=](){
    int &ref = global_host_var;
    ref++;
    *out = ref;
  }();
}

// HOST-LABEL: define{{.*}} void @_Z28dev_capture_host_ref_by_copyPi(
// HOST: %[[CAP:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %{{.*}}, i32 0, i32 1
// HOST: %[[VAL:.*]] = load i32, ptr @global_host_var
// HOST: store i32 %[[VAL]], ptr %[[CAP]]
// CIR-DEV-LABEL: cir.func {{.*}} @_ZZ28dev_capture_host_ref_by_copyPiENKUlvE_clEv(
// CIR-DEV-NOT:     cir.global_view<@global_host_var>
// CIR-DEV:         %[[CAP:.*]] = cir.get_member %{{.*}}[1]
// CIR-DEV:         %[[VAL:.*]] = cir.load {{.*}}%[[CAP]] : !cir.ptr<!s32i>, !s32i
// CIR-DEV:         cir.store {{.*}}%[[VAL]]
// DEV-LABEL: define internal void @_ZZ28dev_capture_host_ref_by_copyPiENKUlvE_clEv(
// DEV: %[[CAP:.*]] = getelementptr inbounds nuw %{{.*}}, ptr %{{.*}}, i32 0, i32 1
// DEV: %[[VAL:.*]] = load i32, ptr %[[CAP]]
// DEV: store i32 %[[VAL]]
void dev_capture_host_ref_by_copy(int *out) {
  int &ref = global_host_var;
  kern<<<1, 1>>>([=]__device__() { *out = ref;});
}
