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

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// Disables RTSan error reporting for the current thread.
///
/// Calls may be nested. Each call must be paired with a subsequent call to
/// \ref __rtsan_enable on the same thread.
void SANITIZER_CDECL __rtsan_disable(void);

/// Re-enables RTSan error reporting for the current thread.
///
/// This function ends the innermost disabled region started by
/// \ref __rtsan_disable. Calling this function without a matching call to
/// \ref __rtsan_disable is invalid.
void SANITIZER_CDECL __rtsan_enable(void);

#ifdef __cplusplus
} // extern "C"

namespace __rtsan {
#if defined(__has_feature) && __has_feature(realtime_sanitizer)

/// Disables RTSan error reporting for the lifetime of an instance.
///
/// Instances may be nested and must be destroyed on the thread on which they
/// were constructed. Instances cannot be copied or moved.
class ScopedDisabler {
public:
  /// Disables RTSan error reporting for the current thread.
  ScopedDisabler() { __rtsan_disable(); }

  /// Re-enables RTSan error reporting for the current thread.
  ~ScopedDisabler() { __rtsan_enable(); }

  /// \cond
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
  /// \endcond
};

#else

class ScopedDisabler {
public:
  ScopedDisabler() {}
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

#endif // defined(__has_feature) && __has_feature(realtime_sanitizer)
} // namespace __rtsan
#endif // __cplusplus

#endif // SANITIZER_RTSAN_INTERFACE_H
