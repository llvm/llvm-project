//===-------------- Simple checks for cross compilation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_CROSS_COMPILE_CLIPPY_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_CROSS_COMPILE_CLIPPY_H

#include "src/__support/macros/properties/architectures.h"
#include <sys/syscall.h>

// https://github.com/hrw/syscalls-table is super helpful for trying to find
// syscalls with unique numbers.

// As of Linux 6.12.10, 32b RISCV does not define __NR_iodestroy.
#if (defined(LIBC_TARGET_ARCH_IS_AARCH64) && (__NR_renameat) != 38) ||         \
    (defined(LIBC_TARGET_ARCH_IS_ARM) && (__NR_renameat) != 329) ||            \
    (defined(LIBC_TARGET_ARCH_IS_X86_32) && (__NR_renameat) != 302) ||         \
    (defined(LIBC_TARGET_ARCH_IS_X86_64) && (__NR_renameat) != 264) ||         \
    (defined(LIBC_TARGET_ARCH_IS_RISCV64) &&                                   \
     (__NR_riscv_flush_icache) != 259 && (__NR_renameat2) != 276) ||           \
    (defined(LIBC_TARGET_ARCH_IS_RISCV32) &&                                   \
     (__NR_riscv_flush_icache) != 259 && !defined(__NR_iodestroy))

// This is bad because the syscall numbers frequently (but not always) differ
// between architectures.  What frequently happens as a result are crashes in
// startup.
#error "Host kernel headers cannot be used to cross compile"

#endif

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_CROSS_COMPILE_CLIPPY_H
