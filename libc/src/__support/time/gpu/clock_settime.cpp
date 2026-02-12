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
#include "src/__support/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<int> clock_settime(clockid_t clockid, const timespec *ts) {
  // GPU hardware clocks are read-only; setting is not supported.
  (void)clockid;
  (void)ts;
  return -1;
}
} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
