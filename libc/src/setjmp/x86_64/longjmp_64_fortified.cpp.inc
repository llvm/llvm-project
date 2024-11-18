//===-- Implementation of longjmp (64-bit) --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/longjmp.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64) || !LIBC_COPT_SETJMP_FORTIFICATION
#error "Invalid file include"
#endif

#include "src/setjmp/checksum.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf, int)) {
  asm(R"(
      mov %[cookie], %%rdx

      mov %c[rbx](%%rdi), %%rbx
      xor %%rbx, %%rdx
      xor %[mask], %%rbx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %c[rbp](%%rdi), %%rbp
      xor %%rbp, %%rdx
      xor %[mask], %%rbp
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %c[r12](%%rdi), %%r12
      xor %%r12, %%rdx
      xor %[mask], %%r12
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %c[r13](%%rdi), %%r13
      xor %%r13, %%rdx
      xor %[mask], %%r13
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %c[r14](%%rdi), %%r14
      xor %%r14, %%rdx
      xor %[mask], %%r14
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %c[r15](%%rdi), %%r15
      xor %%r15, %%rdx
      xor %[mask], %%r15
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %c[rsp](%%rdi), %%rsp
      xor %%rsp, %%rdx
      xor %[mask], %%rsp
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %c[rip](%%rdi), %%rcx
      xor %%rcx, %%rdx
      xor %[mask], %%rcx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      cmp %c[chksum](%%rdi), %%rdx
      jne __libc_jmpbuf_corruption

      cmpl $0x1, %%esi
      adcl $0x0, %%esi
      movq %%rsi, %%rax

      jmpq *%%rcx
      )"
      : [cookie] "=m"(jmpbuf::checksum_cookie), [mask] "=m"(jmpbuf::value_mask)
      :
      [rbx] "i"(offsetof(__jmp_buf, rbx)), [rbp] "i"(offsetof(__jmp_buf, rbp)),
      [r12] "i"(offsetof(__jmp_buf, r12)), [r13] "i"(offsetof(__jmp_buf, r13)),
      [r14] "i"(offsetof(__jmp_buf, r14)), [r15] "i"(offsetof(__jmp_buf, r15)),
      [rsp] "i"(offsetof(__jmp_buf, rsp)), [rip] "i"(offsetof(__jmp_buf, rip)),
      [chksum] "i"(offsetof(__jmp_buf, __chksum)),
      [multiple] "i"(jmpbuf::MULTIPLE)
      : "rax", "rcx", "rdx");
}

} // namespace LIBC_NAMESPACE_DECL
