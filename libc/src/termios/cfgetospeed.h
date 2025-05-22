//===-- Implementation header for cfgetospeed -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TERMIOS_CFGETOSPEED_H
#define LLVM_LIBC_SRC_TERMIOS_CFGETOSPEED_H

#include <termios.h>

namespace LIBC_NAMESPACE {

speed_t cfgetospeed(const struct termios *t);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_TERMIOS_CFGETOSPEED_H
