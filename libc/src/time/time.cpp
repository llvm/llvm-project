//===-- Linux implementation of the time function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/time_func.h"

#include "hdr/time_macros.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/clock_gettime.h"

namespace LIBC_NAMESPACE_DECL {
// avoid inconsitent clang-format behavior
using time_ptr_t = time_t *;
LLVM_LIBC_FUNCTION(time_t, time, (time_ptr_t tp)) {
  struct timespec ts;
  auto result = internal::clock_gettime(CLOCK_REALTIME, &ts);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  if (tp != nullptr)
    *tp = time_t(ts.tv_sec);
  return time_t(ts.tv_sec);
}

} // namespace LIBC_NAMESPACE_DECL
