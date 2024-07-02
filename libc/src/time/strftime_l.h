//===-- Implementation header of strftime_l ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_STRFTIME_L_H
#define LLVM_LIBC_SRC_TIME_STRFTIME_L_H

#include <time.h>
#include "include/llvm-libc-types/locale_t.h"
#include "include/llvm-libc-types/size_t.h"

namespace LIBC_NAMESPACE {

size_t strftime_l(char*__restrict s, size_t max,
                       const char *__restrict format,
                       const struct tm *__restrict tm,
                       locale_t locale);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_TIME_STRFTIME_L_H
