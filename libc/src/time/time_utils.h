//===-- Collection of utils for mktime and friends --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TIME_UTILS_H
#define LLVM_LIBC_SRC_TIME_TIME_UTILS_H

#include "hdr/types/size_t.h"
#include "hdr/types/struct_tm.h"
#include "hdr/types/time_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "time_constants.h"
#include <stdint.h>

#ifdef LIBC_TARGET_OS_IS_LINUX

#include "src/time/linux/localtime_utils.h"
#include "src/time/linux/timezone.h"

#endif

namespace LIBC_NAMESPACE_DECL {
namespace time_utils {

LIBC_INLINE volatile int file_usage;

// Update the "tm" structure's year, month, etc. members from seconds.
// "total_seconds" is the number of seconds since January 1st, 1970.
extern int64_t update_from_seconds(int64_t total_seconds, struct tm *tm);
extern ErrorOr<File *> acquire_file(char *filename);
extern void release_file(ErrorOr<File *> error_or_file);
extern unsigned char is_dst(struct tm *tm);

#ifdef LIBC_TARGET_OS_IS_LINUX
extern char *get_env_var(const char *var_name);
#endif

// TODO(michaelrj): move these functions to use ErrorOr instead of setting
// errno. They always accompany a specific return value so we only need the one
// variable.

// POSIX.1-2017 requires this.
LIBC_INLINE time_t out_of_range() {
#ifdef EOVERFLOW
  // For non-POSIX uses of the standard C time functions, where EOVERFLOW is
  // not defined, it's OK not to set errno at all. The plain C standard doesn't
  // require it.
  libc_errno = EOVERFLOW;
#endif
  return time_constants::OUT_OF_RANGE_RETURN_VALUE;
}

LIBC_INLINE void invalid_value() { libc_errno = EINVAL; }

LIBC_INLINE char *asctime(const struct tm *timeptr, char *buffer,
                          size_t bufferLength) {
  if (timeptr == nullptr || buffer == nullptr) {
    invalid_value();
    return nullptr;
  }
  if (timeptr->tm_wday < 0 ||
      timeptr->tm_wday > (time_constants::DAYS_PER_WEEK - 1)) {
    invalid_value();
    return nullptr;
  }
  if (timeptr->tm_mon < 0 ||
      timeptr->tm_mon > (time_constants::MONTHS_PER_YEAR - 1)) {
    invalid_value();
    return nullptr;
  }

  // TODO(michaelr): move this to use the strftime machinery
  int written_size = __builtin_snprintf(
      buffer, bufferLength, "%.3s %.3s%3d %.2d:%.2d:%.2d %d\n",
      time_constants::WEEK_DAY_NAMES[timeptr->tm_wday].data(),
      time_constants::MONTH_NAMES[timeptr->tm_mon].data(), timeptr->tm_mday,
      timeptr->tm_hour, timeptr->tm_min, timeptr->tm_sec,
      time_constants::TIME_YEAR_BASE + timeptr->tm_year);
  if (written_size < 0)
    return nullptr;
  if (static_cast<size_t>(written_size) >= bufferLength) {
    out_of_range();
    return nullptr;
  }
  return buffer;
}

LIBC_INLINE struct tm *gmtime_internal(const time_t *timer, struct tm *result) {
  int64_t seconds = *timer;
  // Update the tm structure's year, month, day, etc. from seconds.
  if (update_from_seconds(seconds, result) < 0) {
    out_of_range();
    return nullptr;
  }

  return result;
}

LIBC_INLINE struct tm *localtime_internal(const time_t *timer, struct tm *buf) {
  if (timer == nullptr) {
    invalid_value();
    return nullptr;
  }

  // Update the tm structure's year, month, day, etc. from seconds.
  if (update_from_seconds(static_cast<int64_t>(*timer), buf) < 0) {
    out_of_range();
    return nullptr;
  }

#ifdef LIBC_TARGET_OS_IS_LINUX
  // timezone::tzset *ptr = localtime_utils::get_localtime(buf);
  // buf->tm_hour += ptr->global_offset;
  // buf->tm_isdst = ptr->global_isdst;
#endif

  return buf;
}

// for windows only, implemented on gnu/linux for compatibility reasons
LIBC_INLINE int localtime_s_internal(const time_t *t_ptr, struct tm *input) {
  if (input == NULL)
    return -1;

  if ((*t_ptr < 0 || *t_ptr > cpp::numeric_limits<int64_t>::max()) &&
      input != NULL) {
    // setting values to -1 for compatibility reasons
    // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/localtime-s-localtime32-s-localtime64-s
    input->tm_sec = -1;
    input->tm_min = -1;
    input->tm_hour = -1;
    input->tm_mday = -1;
    input->tm_mon = -1;
    input->tm_year = -1;
    input->tm_wday = -1;
    input->tm_yday = -1;
    input->tm_isdst = -1;

    return -1;
  }

  localtime_internal(t_ptr, input);

  return 0;
}

} // namespace time_utils
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_TIME_UTILS_H
