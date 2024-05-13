// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux && target={{aarch64-.+}}

// Basic test for float registers number are accepted.

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unwind.h>

_Unwind_Reason_Code frame_handler(struct _Unwind_Context *ctx, void *arg) {
  (void)arg;
  Dl_info info = {0, 0, 0, 0};

  // Unwind util the main is reached, above frames depend on the platform and
  // architecture.
  if (dladdr(reinterpret_cast<void *>(_Unwind_GetIP(ctx)), &info) &&
      info.dli_sname && !strcmp("main", info.dli_sname))
    _Exit(0);

  return _URC_NO_REASON;
}

__attribute__((noinline)) void foo() {
  // Provide some CFI directives that instructs the unwinder where given
  // float register is.
#if defined(__aarch64__)
  // DWARF register number for V0-V31 registers are 64-95.
  // Previous value of V0 is saved at offset 0 from CFA.
  asm volatile(".cfi_offset 64, 0");
  // From now on the previous value of register can't be restored anymore.
  asm volatile(".cfi_undefined 65");
  asm volatile(".cfi_undefined 95");
  // Previous value of V2 is in V30.
  asm volatile(".cfi_register  66, 94");
#endif
  _Unwind_Backtrace(frame_handler, NULL);
}

int main() {
  foo();
  return -2;
}
