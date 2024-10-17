//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/checksum.h"
#include "src/setjmp/setjmp_impl.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid file include"
#endif

#if LIBC_COPT_SETJMP_ENABLE_FORTIFICATION
#define LOAD_CHKSUM_STATE_REGISTERS()                                          \
  asm("mov %0, %%rcx\n\t" ::"m"(jmpbuf::value_mask) : "rcx");                  \
  asm("mov %0, %%rdx\n\t" ::"m"(jmpbuf::checksum_cookie) : "rdx");

#define STORE_REG(SRC)                                                         \
  "mov %%" #SRC ", %%rax\n\t"                                                  \
  "xor %%rcx, %%rax\n\t"                                                       \
  "mov %%rax, %c[" #SRC "](%%rdi)\n\t"                                         \
  "mul %%rdx\n\t"                                                              \
  "xor %%rax, %%rdx\n\t"                                                       \
  "rol $%c[rotation], %%rdx\n\t"

#define STORE_RSP()                                                            \
  "lea 8(%%rsp), %%rax\n\t"                                                    \
  "xor %%rcx, %%rax\n\t"                                                       \
  "mov %%rax, %c[rsp](%%rdi)\n\t"                                              \
  "mul %%rdx\n\t"                                                              \
  "xor %%rax, %%rdx\n\t"                                                       \
  "rolq $%c[rotation], %%rdx\n\t"

#define STORE_RIP()                                                            \
  "mov (%%rsp), %%rax\n\t"                                                     \
  "xor %%rcx, %%rax\n\t"                                                       \
  "mov %%rax, %c[rip](%%rdi)\n\t"                                              \
  "mul %%rdx\n\t"                                                              \
  "xor %%rax, %%rdx\n\t"                                                       \
  "rolq $%c[rotation], %%rdx\n\t"

#define STORE_CHECKSUM() "mov %%rdx, %c[chksum](%%rdi)\n\t"
#else
#define LOAD_CHKSUM_STATE_REGISTERS()
#define STORE_REG(SRC) "mov %%" #SRC ", %c[" #SRC "](%%rdi)\n\t"
#define STORE_RSP()                                                            \
  "lea 8(%%rsp), %%rax\n\t"                                                    \
  "mov %%rax, %c[rsp](%%rdi)\n\t"
#define STORE_RIP()                                                            \
  "mov (%%rsp), %%rax\n\t"                                                     \
  "mov %%rax, %c[rip](%%rdi)\n\t"
#define STORE_CHECKSUM()
#endif

namespace LIBC_NAMESPACE_DECL {
[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  LOAD_CHKSUM_STATE_REGISTERS()
  asm(
      // clang-format off
    STORE_REG(rbx)
    STORE_REG(rbp)
    STORE_REG(r12)
    STORE_REG(r13)
    STORE_REG(r14)
    STORE_REG(r15)
    STORE_RSP()
    STORE_RIP()
    STORE_CHECKSUM()
      // clang-format on
      ::[rbx] "i"(offsetof(__jmp_buf, rbx)),
      [rbp] "i"(offsetof(__jmp_buf, rbp)), [r12] "i"(offsetof(__jmp_buf, r12)),
      [r13] "i"(offsetof(__jmp_buf, r13)), [r14] "i"(offsetof(__jmp_buf, r14)),
      [r15] "i"(offsetof(__jmp_buf, r15)), [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip))
#if LIBC_COPT_SETJMP_ENABLE_FORTIFICATION
      // clang-format off
      ,[rotation] "i"(jmpbuf::ROTATION)
      ,[chksum] "i"(offsetof(__jmp_buf, __chksum))
  // clang-format on
#endif
      : "rax", "rdx");

  asm(R"(
    xorl %eax, %eax
    retq
  )");
}
#endif

} // namespace LIBC_NAMESPACE_DECL
