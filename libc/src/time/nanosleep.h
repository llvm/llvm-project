//===-- Implementation header of nanosleep -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_NANOSLEEP_H
#define LLVM_LIBC_SRC_TIME_NANOSLEEP_H

#include <time.h>

namespace __llvm_libc {

int nanosleep(const struct timespec *req, struct timespec *rem);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TIME_NANOSLEEP_H
