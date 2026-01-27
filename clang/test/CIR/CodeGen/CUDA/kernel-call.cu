// Based on clang/test/CodeGenCUDA/kernel-call.cu.
// Tests device stub body emission for CUDA kernels.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -emit-cir %s -I%S/../inputs/ -x cuda -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CUDA-NEW


#include "cuda.h"


// TODO: Test CUDA legacy (< 9.0) when legacy stub body is implemented
// TODO: Test HIP when HIP stub body support is complete

// CUDA-NEW-LABEL: cir.func {{.*}} @_Z21__device_stub__kernelv
// CUDA-NEW: cir.call @__cudaPopCallConfiguration
// CUDA-NEW: cir.call @cudaLaunchKernel
__global__ void kernel() {}
