//===--- clock_gettime internal implementation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_TIME_CLOCK_GETTIME_H
#define LLVM_LIBC_SRC___SUPPORT_TIME_CLOCK_GETTIME_H

#ifdef __linux__
#include "src/__support/time/linux/clock_gettime.h"
#else
#error "clock_gettime is not supported on this platform"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_TIME_CLOCK_GETTIME_H
