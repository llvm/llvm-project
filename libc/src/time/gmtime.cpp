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

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(struct tm *, gmtime, (const time_t *timer)) {
  static struct tm tm_out;
  return time_utils::gmtime_internal(timer, &tm_out);
}

} // namespace LIBC_NAMESPACE
