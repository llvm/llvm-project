//===-- Linux implementation of getpid ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpid.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/getpid.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(pid_t, getpid, ()) { return linux_syscalls::getpid(); }

} // namespace LIBC_NAMESPACE_DECL
