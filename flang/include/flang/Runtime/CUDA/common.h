//===-- include/flang/Runtime/CUDA/common.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_COMMON_H_
#define FORTRAN_RUNTIME_CUDA_COMMON_H_

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Runtime/descriptor.h"
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
    Terminator terminator{__FILE__, __LINE__}; \
    terminator.Crash("'%s' failed with '%s'", #expr, name); \
  }(expr)

static inline unsigned getMemType(cuf::DataAttribute attr) {
  if (attr == cuf::DataAttribute::Device)
    return kMemTypeDevice;
  if (attr == cuf::DataAttribute::Managed)
    return kMemTypeManaged;
  if (attr == cuf::DataAttribute::Unified)
    return kMemTypeUnified;
  if (attr == cuf::DataAttribute::Pinned)
    return kMemTypePinned;
  llvm::report_fatal_error("unsupported memory type");
}

#endif // FORTRAN_RUNTIME_CUDA_COMMON_H_
