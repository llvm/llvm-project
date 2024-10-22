//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf, int)) {
  asm(R"(
      cmpl $0x1, %%esi
      adcl $0x0, %%esi
      movq %%rsi, %%rax

      movq %c[rbx](%%rdi), %%rbx
      movq %c[rbp](%%rdi), %%rbp
      movq %c[r12](%%rdi), %%r12
      movq %c[r13](%%rdi), %%r13
      movq %c[r14](%%rdi), %%r14
      movq %c[r15](%%rdi), %%r15
      movq %c[rsp](%%rdi), %%rsp
      jmpq *%c[rip](%%rdi)
      )" ::[rbx] "i"(offsetof(__jmp_buf, rbx)),
      [rbp] "i"(offsetof(__jmp_buf, rbp)), [r12] "i"(offsetof(__jmp_buf, r12)),
      [r13] "i"(offsetof(__jmp_buf, r13)), [r14] "i"(offsetof(__jmp_buf, r14)),
      [r15] "i"(offsetof(__jmp_buf, r15)), [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip)));
}

} // namespace LIBC_NAMESPACE_DECL
