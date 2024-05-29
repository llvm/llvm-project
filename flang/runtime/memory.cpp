//===-- runtime/memory.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/memory.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/freestanding-tools.h"
#include <cstdlib>

namespace Fortran::runtime {
RT_OFFLOAD_API_GROUP_BEGIN

void *AllocateMemoryOrCrash(const Terminator &terminator, std::size_t bytes) {
  if (void *p{std::malloc(bytes)}) {
    return p;
  }
  if (bytes > 0) {
    terminator.Crash(
        "Fortran runtime internal error: out of memory, needed %zd bytes",
        bytes);
  }
  return nullptr;
}

void *ReallocateMemoryOrCrash(
    const Terminator &terminator, void *ptr, std::size_t newByteSize) {
  if (void *p{Fortran::runtime::realloc(ptr, newByteSize)}) {
    return p;
  }
  if (newByteSize > 0) {
    terminator.Crash("Fortran runtime internal error: memory realloc returned "
                     "null, needed %zd bytes",
        newByteSize);
  }
  return nullptr;
}

void FreeMemory(void *p) { std::free(p); }

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
