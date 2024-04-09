//===-- Implementation of setjmp --------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define paste(ns) _ZN22 ## ns ## 6setjmpEP9__jmp_buf
#define expand(x) paste(x)
#define LIBC_NAMESPACE_setjump expand(LIBC_NAMESPACE)
// aka _ZN22__llvm_libc_19_0_0_git6setjmpEP9__jmp_buf
// aka __llvm_libc_19_0_0_git::setjmp(__jmp_buf*)

// Brittle! Changing the layout of __jmp_buf will break this!
#define RBX_OFFSET 0
#define RBP_OFFSET 8
#define R12_OFFSET 16
#define R13_OFFSET 24
#define R14_OFFSET 32
#define R15_OFFSET 40
#define RSP_OFFSET 48
#define RIP_OFFSET 56

.global setjump
.global LIBC_NAMESPACE_setjump

.type setjump, @function
.type LIBC_NAMESPACE_setjump, @function

setjump:
LIBC_NAMESPACE_setjump:

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

.size setjump, . - setjump
.size LIBC_NAMESPACE_setjump, . - LIBC_NAMESPACE_setjump

.section .note.GNU-stack, "", @progbits
