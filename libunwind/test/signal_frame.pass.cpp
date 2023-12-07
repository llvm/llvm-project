// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that functions marked as signal frames are reported as such.

// TODO: Investigate this failure on Apple
// XFAIL: target={{.+}}-apple-{{.+}}

// TODO: Figure out why this fails with Memory Sanitizer.
// XFAIL: msan

// UNSUPPORTED: libunwind-arm-ehabi

// The AIX assembler does not support CFI directives, which
// are necessary to run this test.
// UNSUPPORTED: target=powerpc{{(64)?}}-ibm-aix

// Windows doesn't generally use CFI directives. However, i686
// mingw targets do use DWARF (where CFI directives are supported).
// UNSUPPORTED: target={{x86_64|arm.*|aarch64}}-{{.*}}-windows-{{.*}}

#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <libunwind.h>

void test() {
  asm(".cfi_signal_frame");
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  assert(unw_step(&cursor) > 0);
  assert(unw_is_signal_frame(&cursor));
}

int main(int, char**) {
  test();
  return 0;
}
