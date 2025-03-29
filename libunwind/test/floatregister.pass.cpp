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

#include <stdlib.h>
#include <string.h>
#include <unwind.h>

// Using __attribute__((section("main_func"))) is ELF specific, but then
// this entire test is marked as requiring Linux, so we should be good.
//
// We don't use dladdr() because on musl it's a no-op when statically linked.
extern char __start_main_func;
extern char __stop_main_func;

_Unwind_Reason_Code frame_handler(struct _Unwind_Context *ctx, void *arg) {
  (void)arg;

  // Unwind until the main is reached, above frames depend on the platform and
  // architecture.
  uintptr_t ip = _Unwind_GetIP(ctx);
  if (ip >= (uintptr_t)&__start_main_func &&
      ip < (uintptr_t)&__stop_main_func) {
    _Exit(0);
  }

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

__attribute__((section("main_func"))) int main() {
  foo();
  return -2;
}
