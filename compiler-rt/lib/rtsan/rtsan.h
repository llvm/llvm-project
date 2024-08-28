//===--- rtsan.h - Realtime Sanitizer ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sanitizer_common/sanitizer_internal_defs.h"

extern "C" {

// Initialise rtsan interceptors.
// A call to this method is added to the preinit array on Linux systems.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_init();

// Initializes rtsan if it has not been initialized yet.
// Used by the RTSan runtime to ensure that rtsan is initialized before any
// other rtsan functions are called.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_ensure_initialized();

SANITIZER_INTERFACE_ATTRIBUTE bool __rtsan_is_initialized();

// Enter real-time context.
// When in a real-time context, RTSan interceptors will error if realtime
// violations are detected. Calls to this method are injected at the code
// generation stage when RTSan is enabled.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_enter();

// Exit the real-time context.
// When not in a real-time context, RTSan interceptors will simply forward
// intercepted method calls to the real methods.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_exit();

// Disable all RTSan error reporting.
// Injected into the code if "nosanitize(realtime)" is on a function.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_off();

// Re-enable all RTSan error reporting.
// The counterpart to `__rtsan_off`.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_on();

SANITIZER_INTERFACE_ATTRIBUTE void
__rtsan_expect_not_realtime(const char *intercepted_function_name);

} // extern "C"
