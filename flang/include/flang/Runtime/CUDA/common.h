//===-- include/flang/Runtime/CUDA/common.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_COMMON_H_
#define FORTRAN_RUNTIME_CUDA_COMMON_H_

#include "flang/Runtime/entry-names.h"

/// Type of memory for allocation/deallocation
static constexpr unsigned kMemTypeDevice = 0;
static constexpr unsigned kMemTypeManaged = 1;
static constexpr unsigned kMemTypeUnified = 2;
static constexpr unsigned kMemTypePinned = 3;

/// Data transfer kinds.
static constexpr unsigned kHostToDevice = 0;
static constexpr unsigned kDeviceToHost = 1;
static constexpr unsigned kDeviceToDevice = 2;

#define CUDA_REPORT_IF_ERROR(expr) \
  [](cudaError_t err) { \
    if (err == cudaSuccess) \
      return; \
    const char *name = cudaGetErrorName(err); \
    if (!name) \
      name = "<unknown>"; \
    Fortran::runtime::Terminator terminator{__FILE__, __LINE__}; \
    terminator.Crash("'%s' failed with '%s'", #expr, name); \
  }(expr)

// Like CUDA_REPORT_IF_ERROR, but tolerates a cudaDeviceReset() or runtime
// teardown having already freed the allocation, so CUF frees stay a no-op.
#define CUDA_REPORT_IF_ERROR_ALLOW_TEARDOWN(expr) \
  [](cudaError_t err) { \
    if (err == cudaSuccess || err == cudaErrorCudartUnloading || \
        err == cudaErrorInvalidValue) \
      return; \
    const char *name = cudaGetErrorName(err); \
    if (!name) \
      name = "<unknown>"; \
    Fortran::runtime::Terminator terminator{__FILE__, __LINE__}; \
    terminator.Crash("'%s' failed with '%s'", #expr, name); \
  }(expr)

// _LOC variant of CUDA_REPORT_IF_ERROR_ALLOW_TEARDOWN.
#define CUDA_REPORT_IF_ERROR_ALLOW_TEARDOWN_LOC(expr, file, line) \
  [](cudaError_t err, const char *sourceFile, int sourceLine) { \
    if (err == cudaSuccess || err == cudaErrorCudartUnloading || \
        err == cudaErrorInvalidValue) \
      return; \
    const char *name = cudaGetErrorName(err); \
    if (!name) \
      name = "<unknown>"; \
    Fortran::runtime::Terminator terminator{sourceFile, sourceLine}; \
    terminator.Crash("'%s' failed with '%s'", #expr, name); \
  }(expr, file, line)

#define CUDA_REPORT_IF_ERROR_LOC(expr, file, line) \
  [](cudaError_t err, const char *sourceFile, int sourceLine) { \
    if (err == cudaSuccess) \
      return; \
    const char *name = cudaGetErrorName(err); \
    if (!name) \
      name = "<unknown>"; \
    Fortran::runtime::Terminator terminator{sourceFile, sourceLine}; \
    terminator.Crash("'%s' failed with '%s'", #expr, name); \
  }(expr, sourceFile, sourceLine)

#endif // FORTRAN_RUNTIME_CUDA_COMMON_H_
