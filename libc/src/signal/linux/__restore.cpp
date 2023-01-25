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

#include "include/sys/syscall.h"
#include "src/__support/OSUtil/syscall.h"

namespace __llvm_libc {

extern "C" void __restore_rt()
    __attribute__((no_sanitize("all"),
                   hidden));

extern "C" void __restore_rt() { __llvm_libc::syscall_impl(SYS_rt_sigreturn); }

} // namespace __llvm_libc
