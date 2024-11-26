//===-- Implementation of setjmp (64-bit) ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64) || !LIBC_COPT_SETJMP_FORTIFICATION
#error "Invalid file include"
#endif

#include "src/setjmp/checksum.h"

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  asm(R"(
      mov %%rbx, %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[rbx](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]
      
      mov %%rbp, %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[rbp](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %%r12, %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[r12](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %%r13, %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[r13](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %%r14, %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[r14](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %%r15, %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[r15](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]

      lea 8(%%rsp), %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[rsp](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov (%%rsp), %%rdx
      xor %[mask], %%rdx
      mov %%rdx, %c[rip](%%rdi)
      xor %%rdx, %[cookie]
      mul %[multiple]
      xor %%rdx, %[cookie]

      mov %[cookie], %c[chksum](%%rdi)

      # from this point, rax does not stand for accumulator but for return value
      xor %%rax, %%rax
      ret)" ::[rbx] "i"(offsetof(__jmp_buf, rbx)),
      [rbp] "i"(offsetof(__jmp_buf, rbp)), [r12] "i"(offsetof(__jmp_buf, r12)),
      [r13] "i"(offsetof(__jmp_buf, r13)), [r14] "i"(offsetof(__jmp_buf, r14)),
      [r15] "i"(offsetof(__jmp_buf, r15)), [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip)),
      [chksum] "i"(offsetof(__jmp_buf, __chksum)),
      [multiple] "c"(jmpbuf::MULTIPLE), [cookie] "a"(jmpbuf::checksum_cookie),
      [mask] "S"(jmpbuf::value_mask)
      :);
}

} // namespace LIBC_NAMESPACE_DECL
