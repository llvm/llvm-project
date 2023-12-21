//===-- Implementation of _start for riscv --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/macros/attributes.h"
#include "startup/linux/do_start.h"

extern "C" [[noreturn]] void _start() {
  using namespace LIBC_NAMESPACE;
  LIBC_INLINE_ASM(".option push\n\t"
                  ".option norelax\n\t"
                  "lla gp, __global_pointer$\n\t"
                  ".option pop\n\t");
  // Fetch the args using the frame pointer.
  app.args = reinterpret_cast<Args *>(
      reinterpret_cast<uintptr_t *>(__builtin_frame_address(0)));
  do_start();
}
