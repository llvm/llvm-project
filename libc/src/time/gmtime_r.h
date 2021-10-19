//===-- Implementation header of gmtime_r -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_GMTIME_R_H
#define LLVM_LIBC_SRC_TIME_GMTIME_R_H

#include <time.h>

namespace __llvm_libc {

struct tm *gmtime_r(const time_t *timer, struct tm *result);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TIME_GMTIME_R_H

#include "include/time.h"
