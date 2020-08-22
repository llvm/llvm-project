//===-- runtime/clock.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implement time measurement intrinsic functions

#include "clock.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <time.h>
// FIXME: windows
#include <sys/time.h>

namespace Fortran::runtime {

void RTNAME(DateAndTime)(char *date, std::size_t dateChars) {
  static constexpr int buffSize{16};
  char buffer[buffSize];
  timeval t;
  time_t timer;
  tm time;

  gettimeofday(&t, nullptr);
  timer = t.tv_sec;
  // TODO windows
  localtime_r(&timer, &time);
  if (date) {
    auto len = strftime(buffer, buffSize, "%Y%m%d", &time);
    auto copyLen = std::min(len, dateChars);
    std::memcpy(date, buffer, copyLen);
    for (auto i{copyLen}; i < dateChars; ++i) {
      date[i] = ' ';
    }
  }
}
} // namespace Fortran::runtime
