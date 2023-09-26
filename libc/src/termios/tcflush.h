//===-- Implementation header for tcflush -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TERMIOS_TCFLUSH_H
#define LLVM_LIBC_SRC_TERMIOS_TCFLUSH_H

#include <termios.h>

namespace LIBC_NAMESPACE {

int tcflush(int fd, int queue_selector);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_TERMIOS_TCFLUSH_H
