//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/common.h"
#include "src/setjmp/setjmp_impl.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  asm("mov %%rbx, %c[rbx](%%rdi)\n\t"
      "mov %%rbp, %c[rbp](%%rdi)\n\t"
      "mov %%r12, %c[r12](%%rdi)\n\t"
      "mov %%r13, %c[r13](%%rdi)\n\t"
      "mov %%r14, %c[r14](%%rdi)\n\t"
      "mov %%r15, %c[r15](%%rdi)\n\t"

      "lea 8(%%rsp), %%rax\n\t"
      "mov %%rax, %c[rsp](%%rdi)\n\t"

      "mov (%%rsp), %%rax\n\t"
      "mov %%rax, %c[rip](%%rdi)\n\t"

      "xorl %%eax, %%eax\n\t"
      "retq" ::[rbx] "i"(offsetof(__jmp_buf, rbx)),
      [rbp] "i"(offsetof(__jmp_buf, rbp)), [r12] "i"(offsetof(__jmp_buf, r12)),
      [r13] "i"(offsetof(__jmp_buf, r13)), [r14] "i"(offsetof(__jmp_buf, r14)),
      [r15] "i"(offsetof(__jmp_buf, r15)), [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip))
      : "rax");
}

} // namespace LIBC_NAMESPACE
