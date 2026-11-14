//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getppid.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/getppid.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/getppid.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(pid_t, getppid, ()) { return linux_syscalls::getppid(); }

} // namespace LIBC_NAMESPACE_DECL
