//===-- Linux implementation of umask -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/umask.h"

#include "hdr/types/mode_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/umask.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(mode_t, umask, (mode_t cmask)) {
  return linux_syscalls::umask(cmask);
}

} // namespace LIBC_NAMESPACE_DECL
