//===-- Implementation header for tcgetattr ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TERMIOS_TCGETATTR_H
#define LLVM_LIBC_SRC_TERMIOS_TCGETATTR_H

#include <termios.h>

namespace __llvm_libc {

int tcgetattr(int fd, struct termios *t);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TERMIOS_TCGETATTR_H
