//===-- Implementation header for cfsetospeed -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TERMIOS_CFSETOSPEED_H
#define LLVM_LIBC_SRC_TERMIOS_CFSETOSPEED_H

#include "src/__support/macros/config.h"
#include <termios.h>

namespace LIBC_NAMESPACE_DECL {

int cfsetospeed(struct termios *t, speed_t speed);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TERMIOS_CFSETOSPEED_H
