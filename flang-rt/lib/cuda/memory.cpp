//===-- lib/cuda/memory.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/memory.h"
#include "flang-rt/runtime/assign-impl.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/memmove-function.h"
#include "flang/Runtime/assign.h"

#include "cuda_runtime.h"

#include <cstddef>
#include <optional>

namespace Fortran::runtime::cuda {

struct Memcpy2DLayout {
  void *base;
  std::size_t widthBytes;
  std::size_t height;
  std::size_t pitchBytes;
};

// Get cudaMemcpy2D layout information for a descriptor that can be represented
// as fixed-pitch rows of widthBytes. Returns nullopt for layouts that need the
// general runtime assignment path.
static std::optional<Memcpy2DLayout> GetMemcpy2DLayout(
    const Descriptor &desc, std::size_t widthBytes) {
  if (desc.rank() == 0 || desc.Elements() == 0) {
    return std::nullopt;
  }
  const auto elemBytes = desc.ElementBytes();
  if (elemBytes == 0 || widthBytes == 0 || widthBytes % elemBytes != 0) {
    return std::nullopt;
  }
  std::size_t contiguousBytes = elemBytes;
  int rowDim = 0;
  while (rowDim < desc.rank()) {
    const auto &dim = desc.GetDimension(rowDim);
    if (dim.Extent() != 1 &&
        (dim.ByteStride() < 0 ||
            static_cast<std::size_t>(dim.ByteStride()) != contiguousBytes)) {
      break;
    }
    contiguousBytes *= dim.Extent();
    ++rowDim;
    if (contiguousBytes == widthBytes) {
      break;
    }
  }
  if (contiguousBytes != widthBytes) {
    return std::nullopt;
  }
  Memcpy2DLayout layout;
  layout.base = desc.raw().base_addr;
  layout.widthBytes = widthBytes;
  layout.height = desc.Elements() * elemBytes / widthBytes;
  if (rowDim == desc.rank()) {
    layout.pitchBytes = widthBytes;
    return layout;
  }
  auto pitch = desc.GetDimension(rowDim).ByteStride();
  if (pitch <= 0 || static_cast<std::size_t>(pitch) < widthBytes) {
    return std::nullopt;
  }
  SubscriptValue expected = pitch;
  for (int j = rowDim; j < desc.rank(); ++j) {
    const auto &dim = desc.GetDimension(j);
    if (dim.Extent() != 1 && dim.ByteStride() != expected) {
      return std::nullopt;
    }
    expected *= dim.Extent();
  }
  layout.pitchBytes = static_cast<std::size_t>(pitch);
  return layout;
}

// Collect candidate row widths from the descriptor's leading contiguous
// dimensions, starting with one element.
static int GetContiguousLeadingBytes(
    const Descriptor &desc, std::size_t *bytes) {
  const auto elemBytes = desc.ElementBytes();
  if (elemBytes == 0) {
    return 0;
  }

  int count = 0;
  bytes[count++] = elemBytes;
  std::size_t contiguousBytes = elemBytes;
  for (int j = 0; j < desc.rank(); ++j) {
    const auto &dim = desc.GetDimension(j);
    if (dim.Extent() != 1 &&
        (dim.ByteStride() < 0 ||
            static_cast<std::size_t>(dim.ByteStride()) != contiguousBytes)) {
      break;
    }
    contiguousBytes *= dim.Extent();
    if (contiguousBytes != bytes[count - 1]) {
      bytes[count++] = contiguousBytes;
    }
  }
  return count;
}

// Choose the largest row width that is contiguous in both descriptors, so
// leading-dimension slices can be copied as wider cudaMemcpy2D rows.
static std::size_t GetMemcpy2DWidthBytes(
    const Descriptor &dst, const Descriptor &src) {
  std::size_t dstBytes[maxRank + 1];
  std::size_t srcBytes[maxRank + 1];
  const int dstCount = GetContiguousLeadingBytes(dst, dstBytes);
  const int srcCount = GetContiguousLeadingBytes(src, srcBytes);
  for (int j = dstCount - 1; j >= 0; --j) {
    for (int k = srcCount - 1; k >= 0; --k) {
      if (dstBytes[j] == srcBytes[k]) {
        return dstBytes[j];
      }
    }
  }
  return 0;
}

// Try to use cudaMemcpy2D for a memcpy of two descriptors, returning true if
// successful. False if the 2D data transfer is not possible.
static bool DoMemcpy2D(const Descriptor &dst, const Descriptor &src,
    cudaMemcpyKind kind, const char *sourceFile, int sourceLine) {
  if (dst.ElementBytes() != src.ElementBytes() ||
      dst.Elements() != src.Elements())
    return false;

  std::size_t widthBytes = GetMemcpy2DWidthBytes(dst, src);
  if (widthBytes == 0) {
    return false;
  }
  auto dstLayout = GetMemcpy2DLayout(dst, widthBytes);
  auto srcLayout = GetMemcpy2DLayout(src, widthBytes);
  if (!dstLayout || !srcLayout) {
    return false;
  }

  CUDA_REPORT_IF_ERROR_LOC(
      cudaMemcpy2D(dstLayout->base, dstLayout->pitchBytes, srcLayout->base,
          srcLayout->pitchBytes, widthBytes, dstLayout->height, kind),
      sourceFile, sourceLine);
  return true;
}

static cudaMemcpyKind GetMemcpyKind(
    unsigned mode, const char *sourceFile, int sourceLine) {
  if (mode == kHostToDevice) {
    return cudaMemcpyHostToDevice;
  } else if (mode == kDeviceToHost) {
    return cudaMemcpyDeviceToHost;
  } else if (mode == kDeviceToDevice) {
    return cudaMemcpyDeviceToDevice;
  }
  Terminator terminator{sourceFile, sourceLine};
  terminator.Crash("host to host copy not supported");
}

extern "C" {

void *RTDEF(CUFMemAlloc)(
    std::size_t bytes, unsigned type, const char *sourceFile, int sourceLine) {
  void *ptr = nullptr;
  bytes = bytes ? bytes : 1;
  if (type == kMemTypeDevice) {
    if (Fortran::runtime::executionEnvironment.cudaDeviceIsManaged) {
      CUDA_REPORT_IF_ERROR_LOC(
          cudaMallocManaged((void **)&ptr, bytes, cudaMemAttachGlobal),
          sourceFile, sourceLine);
    } else {
      CUDA_REPORT_IF_ERROR_LOC(
          cudaMalloc((void **)&ptr, bytes), sourceFile, sourceLine);
    }
  } else if (type == kMemTypeManaged || type == kMemTypeUnified) {
    CUDA_REPORT_IF_ERROR_LOC(
        cudaMallocManaged((void **)&ptr, bytes, cudaMemAttachGlobal),
        sourceFile, sourceLine);
  } else if (type == kMemTypePinned) {
    CUDA_REPORT_IF_ERROR_LOC(
        cudaMallocHost((void **)&ptr, bytes), sourceFile, sourceLine);
  } else {
    Terminator terminator{sourceFile, sourceLine};
    terminator.Crash("unsupported memory type");
  }
  return ptr;
}

void RTDEF(CUFMemFree)(
    void *ptr, unsigned type, const char *sourceFile, int sourceLine) {
  if (!ptr)
    return;
  if (type == kMemTypeDevice || type == kMemTypeManaged ||
      type == kMemTypeUnified) {
    CUDA_REPORT_IF_ERROR_LOC(cudaFree(ptr), sourceFile, sourceLine);
  } else if (type == kMemTypePinned) {
    CUDA_REPORT_IF_ERROR_LOC(cudaFreeHost(ptr), sourceFile, sourceLine);
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
  CUDA_REPORT_IF_ERROR_LOC(
      cudaMemcpy(dst, src, bytes, kind), sourceFile, sourceLine);
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
  // Allocate dst descriptor if not allocated.
  if (!dstDesc->IsAllocated()) {
    dstDesc->ApplyMold(*srcDesc, dstDesc->rank());
    dstDesc->Allocate(/*asyncObject=*/nullptr);
  }
  if ((srcDesc->rank() > 0) && (dstDesc->Elements() <= srcDesc->Elements()) &&
      srcDesc->IsContiguous() && dstDesc->IsContiguous()) {
    // Special case when rhs is bigger than lhs and both are contiguous arrays.
    // In this case we do a simple ptr to ptr transfer with the size of lhs.
    // This is be allowed in the reference compiler and it avoids error
    // triggered in the Assign runtime function used for the main case below.
    RTNAME(CUFDataTransferPtrPtr)(dstDesc->raw().base_addr,
        srcDesc->raw().base_addr, dstDesc->Elements() * dstDesc->ElementBytes(),
        mode, sourceFile, sourceLine);
  } else {
    cudaMemcpyKind kind = GetMemcpyKind(mode, sourceFile, sourceLine);
    // Try to use cudaMemcpy2D first, if it fails, fall back to
    // Fortran::runtime::Assign.
    if (DoMemcpy2D(*dstDesc, *srcDesc, kind, sourceFile, sourceLine)) {
      return;
    }
    Fortran::runtime::Assign(
        *dstDesc, *srcDesc, terminator, MaybeReallocate, memmoveFct);
  }
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
    ((Descriptor *)deviceAddr, dstDesc, sourceFile, sourceLine);
  }
}
}
} // namespace Fortran::runtime::cuda
