//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/checksum.h"
#include "src/stdlib/abort.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {

#if LIBC_COPT_SETJMP_ENABLE_FORTIFICATION
extern "C" [[gnu::cold, noreturn]] void __libc_jmpbuf_corruption() {
  write_to_stderr("invalid checksum detected in longjmp\n");
  abort();
}
#define LOAD_CHKSUM_STATE_REGISTERS()                                          \
  asm("mov %0, %%rcx\n\t" ::"m"(jmpbuf::value_mask) : "rcx");                  \
  asm("mov %0, %%rdx\n\t" ::"m"(jmpbuf::checksum_cookie) : "rdx");

#define RESTORE_REG(DST)                                                       \
  "movq %c[" #DST "](%%rdi), %%rax\n\t"                                        \
  "movq %%rax, %%" #DST "\n\t"                                                 \
  "xor %%rcx, %%" #DST "\n\t"                                                  \
  "mul %%rdx\n\t"                                                              \
  "xor %%rax, %%rdx\n\t"                                                       \
  "rol $%c[rotation], %%rdx\n\t"

#define RESTORE_RIP()                                                          \
  "movq %c[rip](%%rdi), %%rax\n\t"                                             \
  "xor %%rax, %%rcx\n\t"                                                       \
  "mul %%rdx\n\t"                                                              \
  "xor %%rax, %%rdx\n\t"                                                       \
  "rol $%c[rotation], %%rdx\n\t"                                               \
  "cmp %c[chksum](%%rdi), %%rdx\n\t"                                           \
  "jne __libc_jmpbuf_corruption\n\t"                                           \
  "cmpl $0x1, %%esi\n\t"                                                       \
  "adcl $0x0, %%esi\n\t"                                                       \
  "movq %%rsi, %%rax\n\t"                                                      \
  "jmp *%%rcx\n\t"
#else
#define LOAD_CHKSUM_STATE_REGISTERS()
#define RESTORE_REG(DST) "movq %c[" #DST "](%%rdi), %%" #DST "\n\t"
#define RESTORE_RIP()                                                          \
  "cmpl $0x1, %%esi\n\t"                                                       \
  "adcl $0x0, %%esi\n\t"                                                       \
  "movq %%rsi, %%rax\n\t"                                                      \
  "jmpq *%c[rip](%%rdi)\n\t"
#endif

[[gnu::naked]]
LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf, int)) {
  LOAD_CHKSUM_STATE_REGISTERS()
  asm(
      // clang-format off
      RESTORE_REG(rbx)
      RESTORE_REG(rbp)
      RESTORE_REG(r12)
      RESTORE_REG(r13)
      RESTORE_REG(r14)
      RESTORE_REG(r15)
      RESTORE_REG(rsp)
      RESTORE_RIP()
      // clang-format on
      ::[rbx] "i"(offsetof(__jmp_buf, rbx)),
      [rbp] "i"(offsetof(__jmp_buf, rbp)), [r12] "i"(offsetof(__jmp_buf, r12)),
      [r13] "i"(offsetof(__jmp_buf, r13)), [r14] "i"(offsetof(__jmp_buf, r14)),
      [r15] "i"(offsetof(__jmp_buf, r15)), [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip))
#if LIBC_COPT_SETJMP_ENABLE_FORTIFICATION
      // clang-format off
      ,[rotation] "i"(jmpbuf::ROTATION)
      , [chksum] "i"(offsetof(__jmp_buf, __chksum))
  // clang-format on
#endif
      : "rax", "rdx", "rcx", "rsi");
}

} // namespace LIBC_NAMESPACE_DECL
