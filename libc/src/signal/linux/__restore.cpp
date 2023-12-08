//===-- Linux implementation of __restore_rt ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file is implemented separately from sigaction.cpp so that we can
// strongly control the options this file is compiled with. __restore_rt cannot
// make any stack allocations so we must ensure this.

#include "src/__support/OSUtil/syscall.h"

#include <sys/syscall.h>

namespace LIBC_NAMESPACE {

extern "C" void __restore_rt()
    __attribute__((no_sanitize("all"),
                   hidden));

extern "C" void __restore_rt() {
  LIBC_NAMESPACE::syscall_impl<long>(SYS_rt_sigreturn);
}

} // namespace LIBC_NAMESPACE
