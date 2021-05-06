/*===- __clang_openmp_device_functions.h - OpenMP device function declares -===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_OPENMP_DEVICE_FUNCTIONS_H__
#define __CLANG_OPENMP_DEVICE_FUNCTIONS_H__

#ifndef _OPENMP
#error "This file is for OpenMP compilation only."
#endif

#ifdef __NVPTX__
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

#define __CUDA__
#define __OPENMP_NVPTX__

/// Include declarations for libdevice functions.
#include <__clang_cuda_libdevice_declares.h>

/// Provide definitions for these functions.
#include <__clang_cuda_device_functions.h>

#undef __OPENMP_NVPTX__
#undef __CUDA__

#ifdef __cplusplus
} // extern "C"
#endif

#pragma omp end declare variant

#endif // __NVPTX__

#ifdef __AMDGCN__

// __NO_INLINE__ prevents some x86 optimized macro definitions in system headers
#define __NO_INLINE__ 1
#pragma omp begin declare variant match(                                       \
    device = {arch(amdgcn)}, implementation = {extension(match_any)})

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

#define __OPENMP_AMDGCN__

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __private __attribute__((address_space(5)))

/// Include declarations for libdevice functions.
#include <__clang_hip_libdevice_declares.h>

#ifdef __cplusplus
} // extern "C"
#endif

#pragma omp end declare variant

#undef __OPENMP_AMDGCN__
#endif // __AMDGCN__

#endif // __CLANG_OPENMP_DEVICE_FUNCTIONS_H__
