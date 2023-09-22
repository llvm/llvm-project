//===-- Implementation header for tcsendbreak -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TERMIOS_TCSENDBREAK_H
#define LLVM_LIBC_SRC_TERMIOS_TCSENDBREAK_H

#include <termios.h>

namespace __llvm_libc {

int tcsendbreak(int fd, int duration);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TERMIOS_TCSENDBREAK_H
