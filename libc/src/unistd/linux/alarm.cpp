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
#include "src/__support/OSUtil/linux/syscall_wrappers/alarm.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, alarm, (unsigned int seconds)) {
  ErrorOr<unsigned int> ret = linux_syscalls::alarm(seconds);
  if (!ret)
    return 0;
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
