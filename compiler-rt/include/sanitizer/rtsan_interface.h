//===-- sanitizer/rtsan_interface.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of RealtimeSanitizer.
//
// Public interface header.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_RTSAN_INTERFACE_H
#define SANITIZER_RTSAN_INTERFACE_H

#if __has_include(<sanitizer/common_interface_defs.h>)
#include <sanitizer/common_interface_defs.h>
#else
#define SANITIZER_CDECL
#endif // __has_include(<sanitizer/common_interface_defs.h>)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Initializes rtsan if it has not been initialized yet.
// Used by the RTSan runtime to ensure that rtsan is initialized before any
// other rtsan functions are called.
void SANITIZER_CDECL __rtsan_ensure_initialized();

// Enter real-time context.
// When in a real-time context, RTSan interceptors will error if realtime
// violations are detected. Calls to this method are injected at the code
// generation stage when RTSan is enabled.
void SANITIZER_CDECL __rtsan_realtime_enter();

// Exit the real-time context.
// When not in a real-time context, RTSan interceptors will simply forward
// intercepted method calls to the real methods.
void SANITIZER_CDECL __rtsan_realtime_exit();

// Disable all RTSan error reporting.
void SANITIZER_CDECL __rtsan_disable(void);

// Re-enable all RTSan error reporting.
// The counterpart to `__rtsan_disable`.
void SANITIZER_CDECL __rtsan_enable(void);

// Expect that the next call to a function with the given name will not be
// called from a realtime context.
void SANITIZER_CDECL
__rtsan_expect_not_realtime(const char *intercepted_function_name);

#ifdef __cplusplus
} // extern "C"

namespace __rtsan {
#if defined(__has_feature) && __has_feature(realtime_sanitizer)

void Initialize() { __rtsan_ensure_initialized(); }

class ScopedEnabler {
public:
  ScopedEnabler() { __rtsan_realtime_enter(); }
  ~ScopedEnabler() { __rtsan_realtime_exit(); }

#if __cplusplus >= 201103L
  ScopedEnabler(const ScopedEnabler &) = delete;
  ScopedEnabler &operator=(const ScopedEnabler &) = delete;
  ScopedEnabler(ScopedEnabler &&) = delete;
  ScopedEnabler &operator=(ScopedEnabler &&) = delete;
#else
private:
  ScopedEnabler(const ScopedEnabler &);
  ScopedEnabler &operator=(const ScopedEnabler &);
#endif // __cplusplus >= 201103L
};

class ScopedDisabler {
public:
  ScopedDisabler() { __rtsan_disable(); }
  ~ScopedDisabler() { __rtsan_enable(); }

#if __cplusplus >= 201103L
  ScopedDisabler(const ScopedDisabler &) = delete;
  ScopedDisabler &operator=(const ScopedDisabler &) = delete;
  ScopedDisabler(ScopedDisabler &&) = delete;
  ScopedDisabler &operator=(ScopedDisabler &&) = delete;
#else
private:
  ScopedDisabler(const ScopedDisabler &);
  ScopedDisabler &operator=(const ScopedDisabler &);
#endif // __cplusplus >= 201103L
};

#else // doesn't have realtime_sanitizer

void Initialize() {}

class ScopedEnabler {
public:
  ScopedEnabler() {}
};

class ScopedDisabler {
public:
  ScopedDisabler() {}
};

#endif // defined(__has_feature) && __has_feature(realtime_sanitizer)
} // namespace __rtsan
#endif // __cplusplus

#endif // SANITIZER_RTSAN_INTERFACE_H
