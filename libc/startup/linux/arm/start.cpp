//===-- Implementation of _start for arm ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "startup/linux/do_start.h"

extern "C" [[noreturn]] void _start() {
  // TODO: implement me! https://github.com/llvm/llvm-project/issues/96326
  // LIBC_NAMESPACE::app.args = reinterpret_cast<LIBC_NAMESPACE::Args *>(
  //     reinterpret_cast<uintptr_t *>(__builtin_frame_address(0)) + 2);
  LIBC_NAMESPACE::do_start();
}
