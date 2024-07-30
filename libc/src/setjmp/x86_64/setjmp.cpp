//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/checksum.h"
#include "src/setjmp/setjmp_impl.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  register __UINT64_TYPE__ rcx __asm__("rcx");
  // Load cookie
  asm("mov %1, %0\n\t" : "=r"(rcx) : "m"(jmpbuf::register_mangle_cookie));
  // store registers to buffer
  // do not pass any invalid values into registers
#define STORE(REG)                                                             \
  asm("mov %%" #REG ", %%rdx\n\t"                                              \
      "xor %%rdx, %%rcx\n\t"                                                   \
      "mov %%rdx, %c[" #REG                                                    \
      "](%%rdi)\n\t" ::[REG] "i"(offsetof(__jmp_buf, REG))                     \
      : "rdx");

  STORE(rbx);
  STORE(rbp);
  STORE(r12);
  STORE(r13);
  STORE(r14);
  STORE(r15);
  asm(R"(
   lea 8(%%rsp),%%rdx
   xor %%rdx, %%rcx
   mov %%rdx,%c[rsp](%%rdi)
   mov (%%rsp),%%rdx
   xor %%rdx, %%rcx     
   mov %%rdx,%c[rip](%%rdi)
 )" ::[rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip))
      : "rdx");

  // tail call to update checksum
  asm("jmp %P0" : : "i"(jmpbuf::update_checksum));
}

} // namespace LIBC_NAMESPACE_DECL
