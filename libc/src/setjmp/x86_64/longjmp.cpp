//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#if LIBC_COPT_SETJMP_FORTIFICATION
#include "src/setjmp/checksum.h"
#endif

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {

#define CALCULATE_RETURN_VALUE()                                               \
  "cmpl $0x1, %%esi\n\t"                                                       \
  "adcl $0x0, %%esi\n\t"                                                       \
  "movq %%rsi, %%rax\n\t"

#if LIBC_COPT_SETJMP_FORTIFICATION
#include "src/setjmp/x86_64/checksum.def"

// clang-format off
#define RESTORE_REG(DST)                                                       \
  "movq %c[" #DST "](%%rdi), %%rax\n\t"                                        \
  "movq %%rax, %%" #DST "\n\t"                                                 \
  "xor %[mask], %%" #DST "\n\t"                                                \
  ACCUMULATE_CHECKSUM()

#define RESTORE_RIP()                                                          \
  "movq %c[rip](%%rdi), %%rax\n\t"                                             \
  "xor %%rax, %[mask]\n\t"                                                     \
  ACCUMULATE_CHECKSUM()                                                        \
  "cmp %c[__chksum](%%rdi), %%rdx\n\t"                                         \
  "jne __libc_jmpbuf_corruption\n\t"                                           \
  CALCULATE_RETURN_VALUE()                                                     \
  "jmp *%[mask]\n\t"
// clang-format on
#else
#define LOAD_CHKSUM_STATE_REGISTERS()
#define RESTORE_REG(DST) "movq %c[" #DST "](%%rdi), %%" #DST "\n\t"
#define RESTORE_RIP()                                                          \
  CALCULATE_RETURN_VALUE()                                                     \
  "jmpq *%c[rip](%%rdi)\n\t"
#endif

[[gnu::naked]] LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf, int)) {
  // use registers to make sure values propagate correctly across the asm blocks
  [[maybe_unused]] register __UINTPTR_TYPE__ mask asm("rcx");
  [[maybe_unused]] register __UINT64_TYPE__ checksum asm("rdx");

  LOAD_CHKSUM_STATE_REGISTERS()
  asm volatile(
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
      : /* outputs */
#if LIBC_COPT_SETJMP_FORTIFICATION
      [mask] "+r"(mask), [checksum] "+r"(checksum)
#endif
      : /* inputs */
      [rbx] "i"(offsetof(__jmp_buf, rbx)), [rbp] "i"(offsetof(__jmp_buf, rbp)),
      [r12] "i"(offsetof(__jmp_buf, r12)), [r13] "i"(offsetof(__jmp_buf, r13)),
      [r14] "i"(offsetof(__jmp_buf, r14)), [r15] "i"(offsetof(__jmp_buf, r15)),
      [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip))
#if LIBC_COPT_SETJMP_FORTIFICATION
      // clang-format off
      ,[rotation] "i"(jmpbuf::ROTATION)
      ,[__chksum] "i"(offsetof(__jmp_buf, __chksum))
  // clang-format on
#endif
      : "rax", "rsi");
}

} // namespace LIBC_NAMESPACE_DECL
