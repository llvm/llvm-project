//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of x86_64 unmap_and_exit.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_X86_64_UNMAP_AND_EXIT_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_X86_64_UNMAP_AND_EXIT_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For SYS_munmap, SYS_exit

namespace LIBC_NAMESPACE_DECL {

// Unmap the calling thread's stack and exit the thread without touching the
// stack between SYS_munmap and SYS_exit. In the x86_64 System V ABI, the first
// two arguments (addr, size) arrive in %rdi and %rsi, matching the syscall ABI
// for SYS_munmap.
[[gnu::naked, noreturn]] static void unmap_and_exit(void *, size_t) {
  LIBC_INLINE_ASM("movl %[sys_munmap], %%eax\n\t"
                  "syscall\n\t"
                  "xorl %%edi, %%edi\n\t"
                  "movl %[sys_exit], %%eax\n\t"
                  "syscall\n\t"
                  "ud2\n\t" : : [sys_munmap] "i"(SYS_munmap),
                  [sys_exit] "i"(SYS_exit));
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_X86_64_UNMAP_AND_EXIT_H
