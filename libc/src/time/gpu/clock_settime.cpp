//===---------- GPU implementation of the clock_settime function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock_settime.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/clock_settime.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, clock_settime,
                   (clockid_t clockid, const timespec *ts)) {
  ErrorOr<int> result = internal::clock_settime(clockid, ts);
  if (result)
    return result.value();
  return result.error();
}

} // namespace LIBC_NAMESPACE_DECL
