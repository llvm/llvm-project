//===-- Implementation of _start for aarch64 ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "startup/linux/do_start.h"
extern "C" [[noreturn]] void _start() {
  // Skip the Frame Pointer and the Link Register
  // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
  // Section 6.2.3. Note that this only works if the current function
  // is not using any callee-saved registers (x19 to x28). If the
  // function uses such registers, then their value is pushed on to the
  // stack before the frame pointer an link register values. That breaks
  // the assumption that stepping over the frame pointer and link register
  // will take us to the previous stack pointer. That is the reason why the
  // actual business logic of the startup code is pushed into a non-inline
  // function do_start so that this function is free of any stack usage.
  LIBC_NAMESPACE::app.args = reinterpret_cast<LIBC_NAMESPACE::Args *>(
      reinterpret_cast<uintptr_t *>(__builtin_frame_address(0)) + 2);
  LIBC_NAMESPACE::do_start();
}
