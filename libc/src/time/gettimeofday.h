//===-- Implementation header of gettimeofday -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_GETTIMEOFDAY_H
#define LLVM_LIBC_SRC_TIME_GETTIMEOFDAY_H

#include <time.h>

namespace __llvm_libc {

int gettimeofday(struct timeval *tv, void *tz);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TIME_GETTIMEOFDAY_H

#include "include/time.h"
