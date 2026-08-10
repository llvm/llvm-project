//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of usleep.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/usleep.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/nanosleep.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, usleep, (useconds_t usec)) {
  static_assert(sizeof(useconds_t) >= 4, "Avoids overflow in ns computation");
  constexpr useconds_t US_IN_S = 1'000'000;
  struct timespec ts = {usec / US_IN_S, (usec % US_IN_S) * 1000};
  auto result = linux_syscalls::nanosleep(&ts, nullptr);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
