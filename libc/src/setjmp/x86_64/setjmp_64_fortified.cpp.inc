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
      mov %[cookie], %%rdx

      mov %%rbx, %%rax
      xor %[mask], %%rax
      mov %%rax, %c[rbx](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx
      
      mov %%rbp, %%rax
      xor %[mask], %%rax
      mov %%rax, %c[rbp](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %%r12, %%rax
      xor %[mask], %%rax
      mov %%rax, %c[r12](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %%r13, %%rax
      xor %[mask], %%rax
      mov %%rax, %c[r13](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %%r14, %%rax
      xor %[mask], %%rax
      mov %%rax, %c[r14](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %%r15, %%rax
      xor %[mask], %%rax
      mov %%rax, %c[r15](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      lea 8(%%rsp), %%rax
      xor %[mask], %%rax
      mov %%rax, %c[rsp](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov (%%rsp), %%rax
      xor %[mask], %%rax
      mov %%rax, %c[rip](%%rdi)
      xor %%rax, %%rdx
      mov $%c[multiple], %%rax
      mul %%rdx
      xor %%rax, %%rdx

      mov %%rdx, %c[chksum](%%rdi)
      xor %%rax, %%rax
      ret)"
      : [cookie] "=m"(jmpbuf::checksum_cookie), [mask] "=m"(jmpbuf::value_mask)
      :
      [rbx] "i"(offsetof(__jmp_buf, rbx)), [rbp] "i"(offsetof(__jmp_buf, rbp)),
      [r12] "i"(offsetof(__jmp_buf, r12)), [r13] "i"(offsetof(__jmp_buf, r13)),
      [r14] "i"(offsetof(__jmp_buf, r14)), [r15] "i"(offsetof(__jmp_buf, r15)),
      [rsp] "i"(offsetof(__jmp_buf, rsp)), [rip] "i"(offsetof(__jmp_buf, rip)),
      [chksum] "i"(offsetof(__jmp_buf, __chksum)),
      [multiple] "i"(jmpbuf::MULTIPLE)
      : "rax", "rdx");
}

} // namespace LIBC_NAMESPACE_DECL
