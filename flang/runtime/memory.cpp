//===-- runtime/memory.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/memory.h"
#include "terminator.h"
#include <cstdlib>

namespace Fortran::runtime {
RT_OFFLOAD_VAR_GROUP_BEGIN

RT_API_ATTRS void *AllocateMemoryOrCrash(
    const Terminator &terminator, std::size_t bytes) {
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

RT_API_ATTRS void FreeMemory(void *p) { std::free(p); }

RT_OFFLOAD_VAR_GROUP_END
} // namespace Fortran::runtime
