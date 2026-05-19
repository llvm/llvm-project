//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of alarm.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/alarm.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, alarm, (unsigned int seconds)) {
  return static_cast<unsigned int>(
      LIBC_NAMESPACE::syscall_impl<long>(SYS_alarm, seconds));
}

} // namespace LIBC_NAMESPACE_DECL
