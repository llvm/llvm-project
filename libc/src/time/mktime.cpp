//===-- Implementation of mktime function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/mktime.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(time_t, mktime, (struct tm * tm_out)) {
  LIBC_CRASH_ON_NULLPTR(tm_out);

  auto mktime_result = time_utils::mktime_internal(tm_out);
  if (!mktime_result) {
    libc_errno = time_utils::TIME_OVERFLOW;
    return time_constants::OUT_OF_RANGE_RETURN_VALUE;
  }

  time_t seconds = *mktime_result;

  // Update the tm structure's year, month, day, etc. from seconds.
  auto status = time_utils::update_from_seconds(seconds, tm_out);
  if (!status) {
    libc_errno = status.error();
    return time_constants::OUT_OF_RANGE_RETURN_VALUE;
  }

  return seconds;
}

} // namespace LIBC_NAMESPACE_DECL
