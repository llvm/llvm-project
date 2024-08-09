//===-- runtime/CUDA/descriptor.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/allocator.h"

namespace Fortran::runtime::cuda {
extern "C" {
RT_EXT_API_GROUP_BEGIN

Descriptor *RTDEF(CUFAllocDesciptor)(
    std::size_t sizeInBytes, const char *sourceFile, int sourceLine) {
  return reinterpret_cast<Descriptor *>(CUFAllocManaged(sizeInBytes));
}

void RTDEF(CUFFreeDesciptor)(
    Descriptor *desc, const char *sourceFile, int sourceLine) {
  CUFFreeManaged(reinterpret_cast<void *>(desc));
}

RT_EXT_API_GROUP_END
}
} // namespace Fortran::runtime::cuda
