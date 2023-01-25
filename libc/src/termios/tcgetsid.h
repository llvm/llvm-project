//===-- Implementation header for tcgetsid ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TERMIOS_TCGETSID_H
#define LLVM_LIBC_SRC_TERMIOS_TCGETSID_H

#include <termios.h>

namespace __llvm_libc {

pid_t tcgetsid(int fd);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TERMIOS_TCGETSID_H
