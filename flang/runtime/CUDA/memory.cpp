//===-- runtime/CUDA/memory.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/memory.h"
#include "../terminator.h"
#include "flang/Runtime/CUDA/common.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {
extern "C" {

void *RTDEF(CUFMemAlloc)(
    std::size_t bytes, unsigned type, const char *sourceFile, int sourceLine) {
  void *ptr = nullptr;
  if (bytes != 0) {
    if (type == kMemTypeDevice) {
      CUDA_REPORT_IF_ERROR(cudaMalloc((void **)&ptr, bytes));
    } else if (type == kMemTypeManaged || type == kMemTypeUnified) {
      CUDA_REPORT_IF_ERROR(
          cudaMallocManaged((void **)&ptr, bytes, cudaMemAttachGlobal));
    } else if (type == kMemTypePinned) {
      CUDA_REPORT_IF_ERROR(cudaMallocHost((void **)&ptr, bytes));
    } else {
      Terminator terminator{sourceFile, sourceLine};
      terminator.Crash("unsupported memory type");
    }
  }
  return ptr;
}

void RTDEF(CUFMemFree)(
    void *ptr, unsigned type, const char *sourceFile, int sourceLine) {
  if (!ptr)
    return;
  if (type == kMemTypeDevice || type == kMemTypeManaged ||
      type == kMemTypeUnified) {
    CUDA_REPORT_IF_ERROR(cudaFree(ptr));
  } else if (type == kMemTypePinned) {
    CUDA_REPORT_IF_ERROR(cudaFreeHost(ptr));
  } else {
    Terminator terminator{sourceFile, sourceLine};
    terminator.Crash("unsupported memory type");
  }
}

void RTDEF(CUFMemsetDescriptor)(const Descriptor &desc, void *value,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  terminator.Crash("not yet implemented: CUDA data transfer from a scalar "
                   "value to a descriptor");
}

void RTDEF(CUFDataTransferPtrPtr)(void *dst, void *src, std::size_t bytes,
    unsigned mode, const char *sourceFile, int sourceLine) {
  cudaMemcpyKind kind;
  if (mode == kHostToDevice) {
    kind = cudaMemcpyHostToDevice;
  } else if (mode == kDeviceToHost) {
    kind = cudaMemcpyDeviceToHost;
  } else if (mode == kDeviceToDevice) {
    kind = cudaMemcpyDeviceToDevice;
  } else {
    Terminator terminator{sourceFile, sourceLine};
    terminator.Crash("host to host copy not supported");
  }
  // TODO: Use cudaMemcpyAsync when we have support for stream.
  CUDA_REPORT_IF_ERROR(cudaMemcpy(dst, src, bytes, kind));
}

void RTDEF(CUFDataTransferDescPtr)(const Descriptor &desc, void *addr,
    std::size_t bytes, unsigned mode, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  terminator.Crash(
      "not yet implemented: CUDA data transfer from a pointer to a descriptor");
}

void RTDEF(CUFDataTransferPtrDesc)(void *addr, const Descriptor &desc,
    std::size_t bytes, unsigned mode, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  terminator.Crash(
      "not yet implemented: CUDA data transfer from a descriptor to a pointer");
}

void RTDECL(CUFDataTransferDescDesc)(const Descriptor &dstDesc,
    const Descriptor &srcDesc, unsigned mode, const char *sourceFile,
    int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  terminator.Crash(
      "not yet implemented: CUDA data transfer between two descriptors");
}
}
} // namespace Fortran::runtime::cuda
