// Based on clang/test/CodeGenCUDA/kernel-call.cu.
// Tests device stub body emission for CUDA and HIP kernels.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -emit-cir %s -x cuda -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CUDA-NEW

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fhip-new-launch-api \
// RUN:   -x hip -emit-cir %s -o %t.hip.cir
// RUN: FileCheck --input-file=%t.hip.cir %s --check-prefix=HIP-NEW


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
// CUDA-NEW: cir.call @cudaLaunchKernel({{.*}}) : (!cir.ptr<!void>, !rec_dim3, !rec_dim3, !cir.ptr<!cir.ptr<!void>>, !u64i, !cir.ptr<!rec_cudaStream>) -> (!u32i {llvm.noundef})
//
// HIP-NEW: cir.global constant external @_Z6kernelif = #cir.global_view<@_Z21__device_stub__kernelif> : !cir.func<(!s32i, !cir.float)>
// HIP-NEW-LABEL: cir.func {{.*}} @_Z21__device_stub__kernelif
// HIP-NEW: cir.alloca !cir.ptr<!rec_hipStream>, {{.*}} ["stream"]
// HIP-NEW: cir.call @__hipPopCallConfiguration({{.*}}) : (!cir.ptr<!rec_dim3>, !cir.ptr<!rec_dim3>, !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!rec_hipStream>>) -> !s32i
// HIP-NEW: cir.get_global @_Z6kernelif : !cir.ptr<!cir.func<(!s32i, !cir.float)>>
// HIP-NEW: cir.call @hipLaunchKernel({{.*}}) : (!cir.ptr<!void>, !rec_dim3, !rec_dim3, !cir.ptr<!cir.ptr<!void>>, !u64i, !cir.ptr<!rec_hipStream>) -> (!u32i {llvm.noundef})
__global__ void kernel(int x, float y) {}
