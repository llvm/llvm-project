//===--- radsan.h - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#pragma once

#define RADSAN_EXPORT __attribute__((visibility("default")))

extern "C" {

/**
    Initialise radsan interceptors. A call to this method is added to the
    preinit array on Linux systems.

    @warning Do not call this method as a user.
*/
RADSAN_EXPORT void radsan_init();

/** Enter real-time context.

    When in a real-time context, RADSan interceptors will error if realtime
    violations are detected. Calls to this method are injected at the code
    generation stage when RADSan is enabled.

    @warning Do not call this method as a user
*/
RADSAN_EXPORT void radsan_realtime_enter();

/** Exit the real-time context.

    When not in a real-time context, RADSan interceptors will simply forward
    intercepted method calls to the real methods.

    @warning Do not call this method as a user
*/
RADSAN_EXPORT void radsan_realtime_exit();

/** Disable all RADSan error reporting.

    This method might be useful to you if RADSan is presenting you with an error
    for some code you are confident is realtime safe. For example, you might
    know that a mutex is never contested, and that locking it will never block
    on your particular system. Be careful!

    A call to `radsan_off()` MUST be paired with a corresponding `radsan_on()`
    to reactivate interception after the code in question. If you don't, radsan
    will cease to work.

    Example:

        float process (float x) [[clang::nonblocking]] 
        {
            auto const y = 2.0f * x;

            radsan_off();
            i_know_this_method_is_realtime_safe_but_radsan_complains_about_it();
            radsan_on();
        }

*/
RADSAN_EXPORT void radsan_off();

/** Re-enable all RADSan error reporting.

    The counterpart to `radsan_off`. See the description for `radsan_off` for
    details about how to use this method.
*/
RADSAN_EXPORT void radsan_on();
}
