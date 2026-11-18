//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of timegm.
///
//===----------------------------------------------------------------------===//

#include "src/time/timegm.h"
#include "hdr/types/struct_tm.h"
#include "hdr/types/time_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(time_t, timegm, (struct tm * tm)) {
  LIBC_CRASH_ON_NULLPTR(tm);

  // NOTE: Currently mktime_internal interprets tm as UTC because LLVM-libc does
  // not support timezones yet (see issue #121962). Once timezone support is
  // added to mktime, timegm must continue to interpret tm strictly as UTC
  // (ignoring TZ / local timezone and tm_isdst).
  auto seconds = time_utils::mktime_internal(tm);
  if (!seconds) {
    libc_errno = time_utils::TIME_OVERFLOW;
    return time_constants::OUT_OF_RANGE_RETURN_VALUE;
  }

  auto status = time_utils::update_from_seconds(*seconds, tm);
  if (!status.has_value()) {
    libc_errno = status.error();
    return time_constants::OUT_OF_RANGE_RETURN_VALUE;
  }

#if defined(__linux__)
  tm->tm_gmtoff = 0;
  tm->tm_zone = "GMT";
#endif

  return *seconds;
}

} // namespace LIBC_NAMESPACE_DECL
