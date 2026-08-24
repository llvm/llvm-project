//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sleep.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/sleep.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/time_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/nanosleep.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, sleep, (unsigned int seconds)) {
  static_assert(sizeof(unsigned int) <= sizeof(time_t), "Avoids overflow");
  struct timespec req = {seconds, 0};
  struct timespec rem = {};
  ErrorOr<int> result = linux_syscalls::nanosleep(&req, &rem);
  if (!result) {
    // Cast does not lose information as `remaining` cannot be greater than
    // `seconds`.
    return static_cast<unsigned int>(rem.tv_sec);
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
