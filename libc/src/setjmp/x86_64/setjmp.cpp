//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  asm(R"(
      mov %%rbx, %[rbx]
      mov %%rbp, %[rbp]
      mov %%r12, %[r12]
      mov %%r13, %[r13]
      mov %%r14, %[r14]
      mov %%r15, %[r15]

      lea 8(%%rsp), %%rax
      mov %%rax, %[rsp]

      mov (%%rsp), %%rax
      mov %%rax, %[rip]
      )" ::
      [rbx] "m"(buf->rbx),
      [rbp] "m"(buf->rbp),
      [r12] "m"(buf->r12),
      [r13] "m"(buf->r13),
      [r14] "m"(buf->r14),
      [r15] "m"(buf->r15),
      [rsp] "m"(buf->rsp),
      [rip] "m"(buf->rip)
      : "rax");
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
