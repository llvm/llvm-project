//===----- Wrapper for hip_host_overlay.h  for openmp host overrides  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __CLANG_OPENMP_HIP_HOST_OVERLAY_H__
#define __CLANG_OPENMP_HIP_HOST_OVERLAY_H__

// This OpenMP auto-included header is only active for HIP or HIP API
// compilations with OpenMP. It provides host overlay functions for certain
// hip APIs to register memory as course-grain for the OpenMP runtime.
// This registration prevents fails when hip allocated memories
// are used by OpenMP such as in map clauses. The third argument to
// omp_register_coarse_grain tells the API to set the HSA coarse-grain
// memory attribute. But since this was already done by the hip function, the
// arg is set to 0 to prevent dual setting of the HSA coarse-grain attribute.
// Only the OpenMP runtime table tracking coarse-grain memory is updated.

#if defined(_OPENMP) && __has_include(<hip/hip_runtime_api.h>) && (defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__))
#include <hip/hip_runtime_api.h>
#include <omp.h>

// Define Overlays for HIP host functions
__attribute__((weak)) hipError_t _ovlh_hipHostMalloc(void **p, size_t sz, unsigned int flags) {
  hipError_t e = hipHostMalloc(p, sz, flags);
  omp_register_coarse_grain_mem(*p, sz, /*set_attr*/ 0);
  return e;
}
__attribute__((weak)) hipError_t _ovlh_hipMalloc(void **p, size_t sz) {
  hipError_t e = hipMalloc(p, sz);
  omp_register_coarse_grain_mem(*p, sz, /*set_attr*/ 0);
  return e;
}

#pragma omp begin declare variant match(device = {kind(host)})

// Calls to hipHostMalloc will register memory with omp(offload) runtime
// to prevent fails with duplicate hsa registration
__attribute__((weak)) hipError_t hipHostMalloc(void **p, size_t sz, unsigned int flags) {
  return _ovlh_hipHostMalloc(p, sz, flags);
}

// Calls to hipMalloc will register memory with omp runtime
__attribute__((weak)) hipError_t hipMalloc(void **p, size_t sz) { return _ovlh_hipMalloc(p, sz); }
#pragma omp end declare variant

#endif // HIP or HIP API compilation with OpenMP.
#endif // __CLANG_OPENMP_HIP_HOST_OVERLAY_H__
