//===-- Implementation of mktime function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/mktime.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(time_t, mktime, (struct tm * tm_out)) {
  auto mktime_result = time_utils::mktime_internal(tm_out);
  if (!mktime_result)
    return time_utils::out_of_range();

  time_t seconds = *mktime_result;

  // Update the tm structure's year, month, day, etc. from seconds.
  if (time_utils::update_from_seconds(seconds, tm_out) < 0)
    return time_utils::out_of_range();

  return seconds;
}

} // namespace LIBC_NAMESPACE_DECL
