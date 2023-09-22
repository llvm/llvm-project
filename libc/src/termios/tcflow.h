//===-- Implementation header for tcflow ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TERMIOS_TCFLOW_H
#define LLVM_LIBC_SRC_TERMIOS_TCFLOW_H

#include <termios.h>

namespace __llvm_libc {

int tcflow(int fd, int action);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TERMIOS_TCFLOW_H
