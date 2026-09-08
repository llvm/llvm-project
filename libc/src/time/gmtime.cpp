//===-- Implementation of gmtime function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/gmtime.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct tm *, gmtime, (const time_t *timer)) {
  LIBC_CRASH_ON_NULLPTR(timer);

  static struct tm tm_out;
  auto result = time_utils::gmtime_internal(timer, &tm_out);
  if (!result) {
    libc_errno = result.error();
    return nullptr;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
