//===-- common.cc -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "atomic_helpers.h"

namespace scudo {

uptr PageSizeCached;
uptr getPageSize();

uptr getPageSizeSlow() {
  PageSizeCached = getPageSize();
  CHECK_NE(PageSizeCached, 0);
  return PageSizeCached;
}

// Fatal internal map() or unmap() error (potentially OOM related).
void NORETURN dieOnMapUnmapError(bool OutOfMemory) {
  outputRaw("Scudo ERROR: internal map or unmap failure");
  if (OutOfMemory)
    outputRaw(" (OOM)");
  outputRaw("\n");
  die();
}

} // namespace scudo
