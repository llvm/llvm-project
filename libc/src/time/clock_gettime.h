//===-- Implementation header for clock_gettime function --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_CLOCK_GETTIME_H
#define LLVM_LIBC_SRC_TIME_CLOCK_GETTIME_H

#include <time.h>

namespace LIBC_NAMESPACE {

int clock_gettime(clockid_t clockid, struct timespec *tp);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_TIME_CLOCK_GETTIME_H
