//===---------- GPU implementation of the POSIX clock_gettime function ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock_gettime.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/clock_gettime.h"
#include "src/__support/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, clock_gettime, (clockid_t clockid, timespec *ts)) {
  ErrorOr<int> result = internal::clock_gettime(clockid, ts);
  if (result)
    return result.value();
  return result.error();
}

} // namespace LIBC_NAMESPACE_DECL
