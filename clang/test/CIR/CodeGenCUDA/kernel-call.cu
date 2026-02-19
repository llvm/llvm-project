// Based on clang/test/CodeGenCUDA/kernel-call.cu.
// Tests device stub body emission and kernel launch for CUDA.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -emit-cir %s -x cuda -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CUDA-NEW
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-sdk-version=9.2 \
// RUN:   -emit-cir %s -x cuda -fcuda-is-device -o %t.device.cir
// RUN: FileCheck --input-file=%t.device.cir %s --check-prefix=DEVICE

#include "Inputs/cuda.h"


// TODO: Test CUDA legacy (< 9.0) when legacy stub body is implemented
// TODO: Test HIP when HIP stub body support is complete

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
// CUDA-NEW: cir.call @cudaLaunchKernel({{.*}}) : (!cir.ptr<!void>, !rec_dim3, !rec_dim3, !cir.ptr<!cir.ptr<!void>>, !u64i, !cir.ptr<!rec_cudaStream>) -> (!u32i {llvm.noundef})

__global__ void kernel(int x, float y) {}

// ===----------------------------------------------------------------------===
// Kernel launch site checks
// ===----------------------------------------------------------------------===


// Device compilation should not emit main
// DEVICE-NOT: @main

// CUDA-NEW-LABEL: cir.func {{.*}} @main
int main(void) {
  // Check dim3 temporaries are allocated for grid and block dimensions
  // CUDA-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["agg.tmp0"]
  // CUDA-NEW-DAG: cir.alloca !rec_dim3, {{.*}} ["agg.tmp1"]
  //
  // Check dim3 constructors are called for grid and block dimensions
  // CUDA-NEW: cir.call @_ZN4dim3C1Ejjj({{.*}}) : (!cir.ptr<!rec_dim3>, !u32i, !u32i, !u32i) -> ()
  // CUDA-NEW: cir.call @_ZN4dim3C1Ejjj({{.*}}) : (!cir.ptr<!rec_dim3>, !u32i, !u32i, !u32i) -> ()
  //
  // Check default shared memory (0) and null stream are set
  // CUDA-NEW: cir.const #cir.int<0> : !u64i
  // CUDA-NEW: cir.const #cir.ptr<null> : !cir.ptr<!rec_cudaStream>
  //
  // Check __cudaPushCallConfiguration is called with grid, block, shared mem, stream
  // CUDA-NEW: cir.call @__cudaPushCallConfiguration({{.*}}) : (!rec_dim3, !rec_dim3, !u64i, !cir.ptr<!rec_cudaStream>) -> !s32i
  //
  // Check the config result is cast to bool for the conditional
  // CUDA-NEW: cir.cast int_to_bool {{.*}} : !s32i -> !cir.bool
  //
  // Check conditional launch: if config fails (true), skip; else call kernel
  // CUDA-NEW: cir.if %{{.*}} {
  // CUDA-NEW: } else {
  // CUDA-NEW:   cir.const #cir.int<42> : !s32i
  // CUDA-NEW:   cir.const #cir.fp<1.000000e+00> : !cir.float
  // CUDA-NEW:   cir.call @_Z21__device_stub__kernelif({{.*}}) : (!s32i, !cir.float) -> ()
  // CUDA-NEW: }
  kernel<<<1, 1>>>(42, 1.0f);
}
