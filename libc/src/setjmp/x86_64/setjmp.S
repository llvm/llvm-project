//===-- Implementation of setjmp --------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/assembly.h"
#include "src/setjmp/x86_64/setjmp.h"

#define paste(ns) _ZN22 ## ns ## 6setjmpEP9__jmp_buf
#define expand(x) paste(x)
#define LIBC_NAMESPACE_setjmp expand(LIBC_NAMESPACE)
// aka _ZN22__llvm_libc_19_0_0_git6setjmpEP9__jmp_buf
// aka __llvm_libc_19_0_0_git::setjmp(__jmp_buf*)

.global SYMBOL_NAME(setjmp)
.global SYMBOL_NAME(LIBC_NAMESPACE_setjmp)

SYMBOL_IS_FUNC(setjmp)
SYMBOL_IS_FUNC(LIBC_NAMESPACE_setjmp)

SYMBOL_NAME(setjmp):
SYMBOL_NAME(LIBC_NAMESPACE_setjmp):

  mov %rbx, RBX_OFFSET(%rdi)
  mov %rbp, RBP_OFFSET(%rdi)
  mov %r12, R12_OFFSET(%rdi)
  mov %r13, R13_OFFSET(%rdi)
  mov %r14, R14_OFFSET(%rdi)
  mov %r15, R15_OFFSET(%rdi)
  lea 8(%rsp), %rax
  mov %rax, RSP_OFFSET(%rdi)
  mov (%rsp), %rax
  mov %rax, RIP_OFFSET(%rdi)
  xor %eax, %eax
  ret
END_FUNC(setjmp)
END_FUNC(LIBC_NAMESPACE_setjmp)

NO_EXEC_STACK_DIRECTIVE
