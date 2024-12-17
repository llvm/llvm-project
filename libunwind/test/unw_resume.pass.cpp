// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that unw_resume() resumes execution at the stack frame identified by
// cursor.

#include <libunwind.h>

__attribute__((noinline)) void test_unw_resume() {
  unw_context_t context = {0};
  unw_cursor_t cursor = {0};

  unw_getcontext(&context);
  unw_init_local(&cursor, &context);
  unw_step(&cursor);
  unw_resume(&cursor);
}

int main() {
  test_unw_resume();
  return 0;
}
