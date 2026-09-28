//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of getpgrp.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/getpgrp.h"

#include "hdr/types/pid_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getpgid.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(pid_t, getpgrp, ()) {
  auto ret = linux_syscalls::getpgid(0);
  // getpgrp() cannot fail, so we can assume getpgid(0) syscall always
  // succeeds (which is true on Linux in "normal execution", when simulation
  // or SECCOMP is not in the picture).
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
