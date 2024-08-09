//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "include/llvm-libc-types/jmp_buf.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/checksum.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(void, longjmp, (__jmp_buf * buf, int val)) {
  asm(R"(
   pushq %%rbp
   pushq %%rbx
   mov  %%rdi, %%rbp
   mov  %%esi, %%ebx
   subq $8, %%rsp
   call %P0
   addq $8, %%rsp
   mov  %%ebx, %%esi
   mov  %%rbp, %%rdi
   popq %%rbx
   popq %%rbp
 )" ::"i"(jmpbuf::verify)
      : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11");

  register __UINT64_TYPE__ rcx __asm__("rcx");
  // Load cookie
  asm("mov %1, %0\n\t" : "=r"(rcx) : "m"(jmpbuf::register_mangle_cookie));

  // load registers from buffer
  // do not pass any invalid values into registers
#define RECOVER(REG)                                                           \
  register __UINT64_TYPE__ REG __asm__(#REG);                                  \
  asm volatile("mov %c[" #REG "](%%rdi), %%rdx\n\t"                            \
               "xor %%rdx, %1\n\t"                                             \
               "mov %%rdx, %0\n\t"                                             \
               : "=r"(REG)                                                     \
               : "r"(rcx), [REG] "i"(offsetof(__jmp_buf, REG))                 \
               : "rdx");

  RECOVER(rbx);
  RECOVER(rbp);
  RECOVER(r12);
  RECOVER(r13);
  RECOVER(r14);
  RECOVER(r15);
  RECOVER(rsp);

  register int eax __asm__("eax");
  asm volatile(R"(
   xor %0,%0
   cmp $1,%%esi       
   adc %%esi,%0
   mov %c[rip](%%rdi),%%rdx
   xor %%rdx, %%rcx
   jmp *%%rdx
 )"
               : "=r"(eax)
               : [rip] "i"(offsetof(__jmp_buf, rip))
               : "rdx");
}

} // namespace LIBC_NAMESPACE_DECL
