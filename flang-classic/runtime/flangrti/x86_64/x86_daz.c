/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* -Mdaz run-time support.
 * For fortran and C main programs, the compiler will add the address of
 * the support routine to ctors
 */

void
__daz(void)
{
  __asm__("pushq	%rax");
  __asm__("stmxcsr	(%rsp)");
  __asm__("popq	%rax");
  __asm__("orq	$64, %rax");
  __asm__("pushq	%rax");
  __asm__("ldmxcsr	(%rsp)");
  __asm__("popq	%rax");
}

