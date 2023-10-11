//===-- linux_common.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

#if SCUDO_LINUX

#include "common.h"
#include "internal_defs.h"
#include "report.h"
#include "report_linux.h"
#include "string_utils.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

namespace scudo {

// Fatal internal map() error (potentially OOM related).
void NORETURN reportMapError(uptr SizeIfOOM) {
  char Error[128] = "Scudo ERROR: internal map failure\n";
  if (SizeIfOOM) {
    formatString(
        Error, sizeof(Error),
        "Scudo ERROR: internal map failure (NO MEMORY) requesting %zuKB\n",
        SizeIfOOM >> 10);
  }
  reportRawError(Error);
}

void NORETURN reportUnmapError(uptr Addr, uptr Size) {
  char Error[128];
  formatString(Error, sizeof(Error),
               "Scudo ERROR: internal unmap failure (error desc=%s) Addr 0x%zx "
               "Size %zu\n",
               strerror(errno), Addr, Size);
  reportRawError(Error);
}

void NORETURN reportProtectError(uptr Addr, uptr Size, int Prot) {
  char Error[128];
  formatString(
      Error, sizeof(Error),
      "Scudo ERROR: internal protect failure (error desc=%s) Addr 0x%zx "
      "Size %zu Prot %x\n",
      strerror(errno), Addr, Size, Prot);
  reportRawError(Error);
}

} // namespace scudo

#endif // SCUDO_LINUX
