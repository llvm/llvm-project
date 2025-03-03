//===-- Collection of utils for localtime -         -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_LOCALTIME_UTILS_H
#define LLVM_LIBC_SRC_TIME_LOCALTIME_UTILS_H

#include "hdr/types/time_t.h"
#include "src/__support/CPP/limits.h"
#include "src/errno/libc_errno.h"
#include "src/time/linux/timezone.h"

namespace LIBC_NAMESPACE_DECL {
namespace localtime_utils {

extern timezone::tzset *get_localtime(struct tm *tm);

} // namespace localtime_utils
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_LINUX_LOCALTIME_UTILS_H
