//===-- Implementation of gmtime function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/gmtime.h"
#include "src/__support/common.h"
#include "src/time/time_utils.h"

#include <limits.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(struct tm *, gmtime, (const time_t *timer)) {
  static struct tm tm_out;
  time_t seconds = *timer;
  // Update the tm structure's year, month, day, etc. from seconds.
  if (time_utils::UpdateFromSeconds(seconds, &tm_out) < 0) {
    time_utils::OutOfRange();
    return nullptr;
  }

  return &tm_out;
}

} // namespace __llvm_libc
