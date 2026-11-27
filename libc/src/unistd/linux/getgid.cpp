//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getgid.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/getgid.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/getgid.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(gid_t, getgid, ()) { return linux_syscalls::getgid(); }

} // namespace LIBC_NAMESPACE_DECL
