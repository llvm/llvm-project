//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation for geteuid.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/geteuid.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/geteuid.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(uid_t, geteuid, ()) { return linux_syscalls::geteuid(); }

} // namespace LIBC_NAMESPACE_DECL
