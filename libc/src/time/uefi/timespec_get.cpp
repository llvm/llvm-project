//===-- Implementation of timespec_get for UEFI ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/timespec_get.h"
#include "hdr/time_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/mktime.h"
#include <Uefi.h>

namespace LIBC_NAMESPACE_DECL {

extern "C" bool __llvm_libc_timespec_get_utc(struct timespec *ts);

LLVM_LIBC_FUNCTION(int, timespec_get, (struct timespec * ts, int base)) {
  if (base != TIME_UTC)
    return 0;

  EFI_TIME efi_time;
  EFI_STATUS status =
      efi_system_table->RuntimeServices->GetTime(&efi_time, nullptr);
  if (status != 0)
    return 0;

  struct tm t;
  t.tm_year = efi_time.Year - 1900; // tm_year is years since 1900
  t.tm_mon = efi_time.Month - 1;    // tm_mon is 0-11
  t.tm_mday = efi_time.Day;
  t.tm_hour = efi_time.Hour;
  t.tm_min = efi_time.Minute;
  t.tm_sec = efi_time.Second;
  t.tm_isdst = -1; // Use system timezone settings

  ts->tv_sec = LIBC_NAMESPACE::mktime(&t);
  if (ts->tv_sec == -1)
    return 0;

  ts->tv_nsec = efi_time.Nanosecond;

  // Handle timezone offset if specified
  if (efi_time.TimeZone != 2047) {        // 2047 means timezone is unspecified
    ts->tv_sec -= efi_time.TimeZone * 60; // EFI_TIME timezone is in minutes
  }
  return base;
}

} // namespace LIBC_NAMESPACE_DECL
