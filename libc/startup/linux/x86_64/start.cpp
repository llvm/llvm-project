//===-- Implementation of _start for x86_64 -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/macros/attributes.h"
#include "startup/linux/do_start.h"

extern "C" [[noreturn]] void _start() {
  // This TU is compiled with -fno-omit-frame-pointer. Hence, the previous
  // value of the base pointer is pushed on to the stack. So, we step over
  // it (the "+ 1" below) to get to the args.
  LIBC_NAMESPACE::app.args = reinterpret_cast<LIBC_NAMESPACE::Args *>(
      reinterpret_cast<uintptr_t *>(__builtin_frame_address(0)) + 1);

  // The x86_64 ABI requires that the stack pointer is aligned to a 16-byte
  // boundary. We align it here but we cannot use any local variables created
  // before the following alignment. Best would be to not create any local
  // variables before the alignment. Also, note that we are aligning the stack
  // downwards as the x86_64 stack grows downwards. This ensures that we don't
  // tread on argc, argv etc.
  // NOTE: Compiler attributes for alignment do not help here as the stack
  // pointer on entry to this _start function is controlled by the OS. In fact,
  // compilers can generate code assuming the alignment as required by the ABI.
  // If the stack pointers as setup by the OS are already aligned, then the
  // following code is a NOP.
  asm volatile("andq $0xfffffffffffffff0, %rsp\n\t");
  asm volatile("andq $0xfffffffffffffff0, %rbp\n\t");

  LIBC_NAMESPACE::do_start();
}
