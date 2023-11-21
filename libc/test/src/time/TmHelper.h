//===---- TmHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_TIME_TM_HELPER_H
#define LLVM_LIBC_TEST_SRC_TIME_TM_HELPER_H

#include <time.h>

#include "src/time/time_utils.h"

using LIBC_NAMESPACE::time_utils::TimeConstants;

namespace LIBC_NAMESPACE {
namespace tmhelper {
namespace testing {

// A helper function to initialize tm data structure.
static inline void initialize_tm_data(struct tm *tm_data, int year, int month,
                                      int mday, int hour, int min, int sec,
                                      int wday, int yday) {
  struct tm temp = {.tm_sec = sec,
                    .tm_min = min,
                    .tm_hour = hour,
                    .tm_mday = mday,
                    .tm_mon = month - 1, // tm_mon starts with 0 for Jan
                    // years since 1900
                    .tm_year = year - TimeConstants::TIME_YEAR_BASE,
                    .tm_wday = wday,
                    .tm_yday = yday,
                    .tm_isdst = 0};
  *tm_data = temp;
}

} // namespace testing
} // namespace tmhelper
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_TEST_SRC_TIME_TM_HELPER_H
