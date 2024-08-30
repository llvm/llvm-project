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

// See documentation in rtsan_interface.h.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_ensure_initialized();

SANITIZER_INTERFACE_ATTRIBUTE bool __rtsan_is_initialized();

// See documentation in rtsan_interface.h.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_enter();

// See documentation in rtsan_interface.h.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_realtime_exit();

// See documentation in rtsan_interface.h.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_disable();

// See documentation in rtsan_interface.h.
SANITIZER_INTERFACE_ATTRIBUTE void __rtsan_enable();

// See documentation in rtsan_interface.h.
SANITIZER_INTERFACE_ATTRIBUTE void
__rtsan_expect_not_realtime(const char *intercepted_function_name);

} // extern "C"
