// Based on clang/test/CodeGenCUDA/kernel-call.cu.
// Tests device stub body emission and kernel launch for CUDA/HIP.


// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -emit-cir %s -x cuda -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CUDA-NEW

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fhip-new-launch-api \
// RUN:   -x hip -emit-cir %s -o %t.hip.cir
// RUN: FileCheck --input-file=%t.hip.cir %s --check-prefix=HIP-NEW

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-sdk-version=9.2 \
// RUN:   -emit-cir %s -x cuda -fcuda-is-device -o %t.device.cir
// RUN: FileCheck --input-file=%t.device.cir %s --check-prefix=DEVICE

#include "Inputs/cuda.h"


// TODO: Test CUDA legacy (< 9.0) when legacy stub body is implemented

// Check that the stub function is generated with the correct name
// CUDA-NEW-LABEL: cir.func {{.*}} @_Z21__device_stub__kernelif
//
// Check kernel arguments are allocated as local variables
// CUDA-NEW-DAG: cir.alloca !s32i, {{.*}} ["x", init]
// CUDA-NEW-DAG: cir.alloca !cir.float, {{.*}} ["y", init]
//
// Check void *args[] array is created with correct size (2 args)
// CUDA-NEW: cir.alloca !cir.array<!cir.ptr<!void> x 2>, {{.*}} ["kernel_args"]
// CUDA-NEW: cir.cast array_to_ptrdecay
//
// Check arguments are stored in the args array via ptr_stride indexing
// CUDA-NEW: cir.const #cir.int<0>
// CUDA-NEW: cir.ptr_stride
// CUDA-NEW: cir.cast bitcast {{.*}} -> !cir.ptr<!void>
// CUDA-NEW: cir.store {{.*}} !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CUDA-NEW: cir.const #cir.int<1>
// CUDA-NEW: cir.ptr_stride
// CUDA-NEW: cir.cast bitcast {{.*}} -> !cir.ptr<!void>
// CUDA-NEW: cir.store {{.*}} !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
//
// Check dim3 grid_dim and block_dim allocas for launch configuration
// CUDA-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["grid_dim"]
// CUDA-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["block_dim"]
//
// Check shared_mem (size_t) and stream allocas
// CUDA-NEW-DAG: cir.alloca !u64i, {{.*}} ["shared_mem"]
// CUDA-NEW-DAG: cir.alloca !cir.ptr<!rec_cudaStream>, {{.*}} ["stream"]
//
// Check __cudaPopCallConfiguration is called with correct argument types
// CUDA-NEW: cir.call @__cudaPopCallConfiguration({{.*}}) : (!cir.ptr<!rec_dim3>, !cir.ptr<!rec_dim3>, !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!rec_cudaStream>>) -> !s32i

//
// Check cudaLaunchKernel is called with all 6 arguments:
// func ptr, gridDim, blockDim, args, sharedMem, stream
// CUDA-NEW: cir.call @cudaLaunchKernel({{.*}}) : (!cir.ptr<!void>{{.*}}, !rec_dim3, !rec_dim3, !cir.ptr<!cir.ptr<!void>>{{.*}}, !u64i{{.*}}, !cir.ptr<!rec_cudaStream>{{.*}}) -> (!u32i {llvm.noundef})
//
// HIP-NEW: cir.global constant external @_Z6kernelif = #cir.global_view<@_Z21__device_stub__kernelif> : !cir.func<(!s32i, !cir.float)>
// HIP-NEW-LABEL: cir.func {{.*}} @_Z21__device_stub__kernelif
// HIP-NEW: cir.alloca !cir.ptr<!rec_hipStream>, {{.*}} ["stream"]
// HIP-NEW: cir.call @__hipPopCallConfiguration({{.*}}) : (!cir.ptr<!rec_dim3>, !cir.ptr<!rec_dim3>, !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!rec_hipStream>>) -> !s32i
// HIP-NEW: cir.get_global @_Z6kernelif : !cir.ptr<!cir.func<(!s32i, !cir.float)>>
// HIP-NEW: cir.call @hipLaunchKernel({{.*}}) : (!cir.ptr<!void> {{.*}}, !rec_dim3, !rec_dim3, !cir.ptr<!cir.ptr<!void>>{{.*}}, !u64i{{.*}}, !cir.ptr<!rec_hipStream>{{.*}}) -> (!u32i {llvm.noundef})
__global__ void kernel(int x, float y) {}

// ===----------------------------------------------------------------------===
// Kernel launch site checks
// ===----------------------------------------------------------------------===


// Device compilation should not emit main
// DEVICE-NOT: @main

// CUDA-NEW-LABEL: cir.func {{.*}} @main
// HIP-NEW-LABEL: cir.func {{.*}} @main
int main(void) {
  // Check dim3 temporaries are allocated for grid and block dimensions
  // CUDA-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["agg.tmp0"]
  // CUDA-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["agg.tmp1"]
  // HIP-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["agg.tmp0"]
  // HIP-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["agg.tmp1"]
  //
  // Check dim3 constructors are called for grid and block dimensions
  // CUDA-NEW: cir.call @_ZN4dim3C1Ejjj({{.*}}) : (!cir.ptr<!rec_dim3> {llvm.align = 4 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}) -> ()
  // CUDA-NEW: cir.call @_ZN4dim3C1Ejjj({{.*}}) : (!cir.ptr<!rec_dim3> {llvm.align = 4 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}) -> ()
  // HIP-NEW: cir.call @_ZN4dim3C1Ejjj({{.*}}) : (!cir.ptr<!rec_dim3> {llvm.align = 4 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}) -> ()
  // HIP-NEW: cir.call @_ZN4dim3C1Ejjj({{.*}}) : (!cir.ptr<!rec_dim3> {llvm.align = 4 : i64, llvm.dereferenceable = 12 : i64, llvm.nonnull, llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}, !u32i {llvm.noundef}) -> ()
  //
  // Check default shared memory (0) and null stream are set
  // CUDA-NEW: cir.const #cir.int<0> : !u64i
  // CUDA-NEW: cir.const #cir.ptr<null> : !cir.ptr<!rec_cudaStream>
  // HIP-NEW: cir.const #cir.int<0> : !u64i
  // HIP-NEW: cir.const #cir.ptr<null> : !cir.ptr<!rec_hipStream>
  //
  // Check Push call configuration is called with grid, block, shared mem, stream
  // CUDA-NEW: cir.call @__cudaPushCallConfiguration({{.*}}) : (!rec_dim3, !rec_dim3, !u64i {llvm.noundef}, !cir.ptr<!rec_cudaStream> {llvm.noundef}) -> !s32i
  // HIP-NEW: cir.call @__hipPushCallConfiguration({{.*}}) : (!rec_dim3, !rec_dim3, !u64i {llvm.noundef}, !cir.ptr<!rec_hipStream> {llvm.noundef}) -> !u32i
  //
  // Check the config result is cast to bool for the conditional
  // CUDA-NEW: cir.cast int_to_bool {{.*}} : !s32i -> !cir.bool
  // HIP-NEW: cir.cast int_to_bool {{.*}} : !u32i -> !cir.bool
  //
  // Check conditional launch: if config fails (true), skip; else call kernel
  // CUDA-NEW: cir.if %{{.*}} {
  // CUDA-NEW: } else {
  // CUDA-NEW:   cir.const #cir.int<42> : !s32i
  // CUDA-NEW:   cir.const #cir.fp<1.000000e+00> : !cir.float
  // CUDA-NEW:   cir.call @_Z21__device_stub__kernelif({{.*}}) {cu.kernel_name = #cir.cu.kernel_name<_Z6kernelif>} : (!s32i {llvm.noundef}, !cir.float {llvm.noundef}) -> ()
  // CUDA-NEW: }
  // HIP-NEW: cir.if %{{.*}} {
  // HIP-NEW: } else {
  // HIP-NEW:   cir.const #cir.int<42> : !s32i
  // HIP-NEW:   cir.const #cir.fp<1.000000e+00> : !cir.float
  // HIP-NEW:   cir.call @_Z21__device_stub__kernelif({{.*}}) {cu.kernel_name = #cir.cu.kernel_name<_Z6kernelif>} : (!s32i {llvm.noundef}, !cir.float {llvm.noundef}) -> ()
  // HIP-NEW: }
  kernel<<<1, 1>>>(42, 1.0f);
}
