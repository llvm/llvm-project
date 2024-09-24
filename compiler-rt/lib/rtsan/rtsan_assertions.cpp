//===--- rtsan_assertions.cpp - Realtime Sanitizer --------------*- C++ -*-===//
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
#include "rtsan/rtsan_assertions.h"

#include "rtsan/rtsan.h"
#include "rtsan/rtsan_diagnostics.h"

using namespace __sanitizer;

void __rtsan::ExpectNotRealtime(Context &context, const DiagnosticsInfo &info) {
  CHECK(__rtsan_is_initialized());
  if (context.InRealtimeContext() && !context.IsBypassed()) {
    context.BypassPush();

    PrintDiagnostics(info);
    Die();
    context.BypassPop();
  }
}
