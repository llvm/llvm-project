//===-- runtime/CUDA/memory.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/memory.h"
#include "../terminator.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {
extern "C" {

void RTDEF(CUFMemsetDescriptor)(const Descriptor &desc, void *value,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  terminator.Crash("not yet implemented: CUDA data transfer from a scalar "
                   "value to a descriptor");
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
