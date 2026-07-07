//===--- Conversion helpers for __kernel_timespec ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_KERNEL_TIMESPEC_H
#define LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_KERNEL_TIMESPEC_H

#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/config.h"
#include <linux/time_types.h>

namespace LIBC_NAMESPACE_DECL {

// Convert from timespec to __kernel_timespec
LIBC_INLINE constexpr __kernel_timespec to_kernel_timespec(const timespec &ts) {
  return __kernel_timespec{
      static_cast<decltype(__kernel_timespec::tv_sec)>(ts.tv_sec),
      static_cast<decltype(__kernel_timespec::tv_nsec)>(ts.tv_nsec)};
}

// Convert from __kernel_timespec to timespec
LIBC_INLINE constexpr timespec to_timespec(const __kernel_timespec &ts64) {
  return timespec{
      static_cast<decltype(timespec::tv_sec)>(ts64.tv_sec),
      static_cast<decltype(timespec::tv_nsec)>(ts64.tv_nsec)};
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_KERNEL_TIMESPEC_H
