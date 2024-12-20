//===-- runtime/CUDA/memory.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/memory.h"
#include "../assign-impl.h"
#include "../terminator.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/memmove-function.h"
#include "flang/Runtime/assign.h"

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

void RTDEF(CUFMemsetDescriptor)(
    Descriptor *desc, void *value, const char *sourceFile, int sourceLine) {
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

void RTDEF(CUFDataTransferPtrDesc)(void *addr, Descriptor *desc,
    std::size_t bytes, unsigned mode, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  terminator.Crash(
      "not yet implemented: CUDA data transfer from a descriptor to a pointer");
}

void RTDECL(CUFDataTransferDescDesc)(Descriptor *dstDesc, Descriptor *srcDesc,
    unsigned mode, const char *sourceFile, int sourceLine) {
  MemmoveFct memmoveFct;
  Terminator terminator{sourceFile, sourceLine};
  if (mode == kHostToDevice) {
    memmoveFct = &MemmoveHostToDevice;
  } else if (mode == kDeviceToHost) {
    memmoveFct = &MemmoveDeviceToHost;
  } else if (mode == kDeviceToDevice) {
    memmoveFct = &MemmoveDeviceToDevice;
  } else {
    terminator.Crash("host to host copy not supported");
  }
  Fortran::runtime::Assign(
      *dstDesc, *srcDesc, terminator, MaybeReallocate, memmoveFct);
}

void RTDECL(CUFDataTransferCstDesc)(Descriptor *dstDesc, Descriptor *srcDesc,
    unsigned mode, const char *sourceFile, int sourceLine) {
  MemmoveFct memmoveFct;
  Terminator terminator{sourceFile, sourceLine};
  if (mode == kHostToDevice) {
    memmoveFct = &MemmoveHostToDevice;
  } else if (mode == kDeviceToHost) {
    memmoveFct = &MemmoveDeviceToHost;
  } else if (mode == kDeviceToDevice) {
    memmoveFct = &MemmoveDeviceToDevice;
  } else {
    terminator.Crash("host to host copy not supported");
  }

  Fortran::runtime::DoFromSourceAssign(
      *dstDesc, *srcDesc, terminator, memmoveFct);
}

void RTDECL(CUFDataTransferDescDescNoRealloc)(Descriptor *dstDesc,
    Descriptor *srcDesc, unsigned mode, const char *sourceFile,
    int sourceLine) {
  MemmoveFct memmoveFct;
  Terminator terminator{sourceFile, sourceLine};
  if (mode == kHostToDevice) {
    memmoveFct = &MemmoveHostToDevice;
  } else if (mode == kDeviceToHost) {
    memmoveFct = &MemmoveDeviceToHost;
  } else if (mode == kDeviceToDevice) {
    memmoveFct = &MemmoveDeviceToDevice;
  } else {
    terminator.Crash("host to host copy not supported");
  }
  Fortran::runtime::Assign(
      *dstDesc, *srcDesc, terminator, NoAssignFlags, memmoveFct);
}

void RTDECL(CUFDataTransferGlobalDescDesc)(Descriptor *dstDesc,
    Descriptor *srcDesc, unsigned mode, const char *sourceFile,
    int sourceLine) {
  RTNAME(CUFDataTransferDescDesc)
  (dstDesc, srcDesc, mode, sourceFile, sourceLine);
  if ((mode == kHostToDevice) || (mode == kDeviceToDevice)) {
    void *deviceAddr{
        RTNAME(CUFGetDeviceAddress)((void *)dstDesc, sourceFile, sourceLine)};
    RTNAME(CUFDescriptorSync)
    ((Descriptor *)deviceAddr, srcDesc, sourceFile, sourceLine);
  }
}
}
} // namespace Fortran::runtime::cuda
