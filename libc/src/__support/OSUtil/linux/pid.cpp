//===------------ pid_t utilities implementation ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/pid.h"
#include "src/__support/OSUtil/syscall.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

pid_t ProcessIdentity::cache = -1;
pid_t ProcessIdentity::get_uncached() {
  return syscall_impl<pid_t>(SYS_getpid);
}

} // namespace LIBC_NAMESPACE_DECL
