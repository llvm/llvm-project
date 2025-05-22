//===-- Implementation header of gmtime -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_GMTIME_H
#define LLVM_LIBC_SRC_TIME_GMTIME_H

#include <time.h>

namespace LIBC_NAMESPACE {

struct tm *gmtime(const time_t *timer);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_TIME_GMTIME_H
