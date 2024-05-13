//===--- timeout linux implementation ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_TIMEOUT_H
#define LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_TIMEOUT_H

#include "hdr/time_macros.h"
#include "src/__support/time/linux/clock_conversion.h"
#include "src/__support/time/linux/timeout_basic.h"

namespace LIBC_NAMESPACE {
namespace internal {
// We use AbsTimeout to remind ourselves that the timeout is an absolute time.
// This is a simple wrapper around the timespec struct that also keeps track of
// whether the time is in realtime or monotonic time.

// This struct is going to be used in timed locks. Pthread generally uses
// realtime clocks for timeouts. However, due to non-monotoncity, realtime
// clocks reportedly lead to undesired behaviors. Therefore, we also provide a
// method to convert the timespec to a monotonic clock relative to the time of
// function call.
LIBC_INLINE void AbsTimeout::ensure_monotonic() {
  if (realtime_flag) {
    timeout = convert_clock(timeout, CLOCK_REALTIME, CLOCK_MONOTONIC);
    realtime_flag = false;
  }
}
} // namespace internal
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_TIMEOUT_H
