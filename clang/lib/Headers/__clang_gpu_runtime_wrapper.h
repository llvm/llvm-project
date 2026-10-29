//===---- __clang_gpu_runtime_wrapper.h - Hermetic HIP/CUDA bootstrap ------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

// Dialect-neutral bootstrap shared by the hermetic HIP and CUDA offload paths.
// It establishes only the device attribute macros the vendor headers expect and
// pulls in no device code.

#ifndef __CLANG_GPU_RUNTIME_WRAPPER_H__
#define __CLANG_GPU_RUNTIME_WRAPPER_H__

#if defined(__HIP__) || defined(__CUDA__)

#ifndef __device__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#endif

#endif // defined(__HIP__) || defined(__CUDA__)
#endif // __CLANG_GPU_RUNTIME_WRAPPER_H__
