//===--------- Linux implementation of the personality function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/personality/personality.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/personality.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, personality, (unsigned long persona)) {
  auto result = linux_syscalls::personality(persona);
  if (!result.has_value()) {
    libc_errno = static_cast<int>(result.error());
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
