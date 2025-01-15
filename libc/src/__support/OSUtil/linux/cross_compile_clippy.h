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

#define MSG "Looks like you may be using the host kernel headers to cross " \
  "compile. This is bad because the syscall numbers frequently (but not " \
  "always) differ between architectures.  What frequently happens as a " \
  "result are crashes in startup."

// https://github.com/hrw/syscalls-table is super helpful for trying to find
// syscalls with unique numbers.

#ifdef LIBC_TARGET_ARCH_IS_AARCH64
static_assert(__NR_renameat == 38, MSG);
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
static_assert(__NR_renameat == 329, MSG);
#elif defined(LIBC_TARGET_ARCH_IS_X86_32)
static_assert(__NR_renameat == 302, MSG);
#elif defined(LIBC_TARGET_ARCH_IS_X86_64)
static_assert(__NR_renameat == 264, MSG);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV64)
static_assert(__NR_riscv_flush_icache == 259, MSG);
static_assert(__NR_renameat2 == 276, MSG);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV32)
static_assert(__NR_riscv_flush_icache == 259, MSG);
#ifdef __NR_iodestroy
#error MSG
#endif
#else
#error "Missing cross compile check for new arch"
#endif


#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_CROSS_COMPILE_CLIPPY_H
