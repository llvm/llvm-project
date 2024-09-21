//===--- rtsan_assertions.h - Realtime Sanitizer ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the RealtimeSanitizer runtime library
//
//===----------------------------------------------------------------------===//

#pragma once

#include "rtsan/rtsan.h"
#include "rtsan/rtsan_context.h"

namespace __rtsan {

template <typename OnViolationAction>
void ExpectNotRealtime(Context &context, OnViolationAction &&on_violation) {
  CHECK(__rtsan_is_initialized());
  if (context.InRealtimeContext() && !context.IsBypassed()) {
    context.BypassPush();
    on_violation();
    context.BypassPop();
  }
}

} // namespace __rtsan
