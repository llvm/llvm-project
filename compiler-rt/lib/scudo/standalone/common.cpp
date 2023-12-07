//===-- common.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "atomic_helpers.h"
#include "string_utils.h"

namespace scudo {

uptr PageSizeCached;
uptr getPageSize();

uptr getPageSizeSlow() {
  PageSizeCached = getPageSize();
  CHECK_NE(PageSizeCached, 0);
  return PageSizeCached;
}

// Fatal internal map() or unmap() error (potentially OOM related).
void NORETURN dieOnMapUnmapError(uptr SizeIfOOM) {
  char Error[128] = "Scudo ERROR: internal map or unmap failure\n";
  if (SizeIfOOM) {
    formatString(
        Error, sizeof(Error),
        "Scudo ERROR: internal map failure (NO MEMORY) requesting %zuKB\n",
        SizeIfOOM >> 10);
  }
  outputRaw(Error);
  setAbortMessage(Error);
  die();
}

} // namespace scudo
