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
      mov %c[rbx](%%rdi), %%rbx
      xor %%rbx, %[cookie]
      xor %[mask], %%rbx
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %c[rbp](%%rdi), %%rbp
      xor %%rbp, %[cookie]
      xor %[mask], %%rbp
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %c[r12](%%rdi), %%r12
      xor %%r12, %[cookie]
      xor %[mask], %%r12
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %c[r13](%%rdi), %%r13
      xor %%r13, %[cookie]
      xor %[mask], %%r13
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %c[r14](%%rdi), %%r14
      xor %%r14, %[cookie]
      xor %[mask], %%r14
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %c[r15](%%rdi), %%r15
      xor %%r15, %[cookie]
      xor %[mask], %%r15
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %c[rsp](%%rdi), %%rsp
      xor %%rsp, %[cookie]
      xor %[mask], %%rsp
      mul %[multiple]
      xor %%rdx, %[cookie]

      # move multiplication factor (which should be in rcx) to rdx
      # free up rcx for PC recovery
      mov %[multiple], %%rdx
      mov %c[rip](%%rdi), %%rcx
      xor %%rcx, %[cookie]
      xor %[mask], %%rcx
      mul %%rdx
      xor %%rdx, %[cookie]

      cmp %c[chksum](%%rdi), %[cookie]
      jne __libc_jmpbuf_corruption

      # from this point, rax does not stand for accumulator but for return value 
      cmpl $0x1, %%esi
      adcl $0x0, %%esi
      movq %%rsi, %%rax 

      jmpq *%%rcx
      )" ::[rbx] "i"(offsetof(__jmp_buf, rbx)),
      [rbp] "i"(offsetof(__jmp_buf, rbp)), [r12] "i"(offsetof(__jmp_buf, r12)),
      [r13] "i"(offsetof(__jmp_buf, r13)), [r14] "i"(offsetof(__jmp_buf, r14)),
      [r15] "i"(offsetof(__jmp_buf, r15)), [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip)),
      [chksum] "i"(offsetof(__jmp_buf, __chksum)),
      [multiple] "c"(jmpbuf::MULTIPLE), [cookie] "a"(jmpbuf::checksum_cookie),
      [mask] "m"(jmpbuf::value_mask)
      :);
}

} // namespace LIBC_NAMESPACE_DECL
