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
#include "src/setjmp/setjmp_impl.h"

#if LIBC_COPT_SETJMP_FORTIFICATION
#include "src/setjmp/checksum.h"
#endif

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif

#if LIBC_COPT_SETJMP_FORTIFICATION
#include "src/setjmp/x86_64/checksum.def"

#define STORE_REG(SRC)                                                         \
  "mov %%" #SRC ", %%rax\n\t"                                                  \
  "xor %[mask], %%rax\n\t"                                                     \
  "mov %%rax, %c[" #SRC "](%%rdi)\n\t" ACCUMULATE_CHECKSUM()

#define STORE_RSP()                                                            \
  "lea 8(%%rsp), %%rax\n\t"                                                    \
  "xor %[mask], %%rax\n\t"                                                     \
  "mov %%rax, %c[rsp](%%rdi)\n\t" ACCUMULATE_CHECKSUM()

#define STORE_RIP()                                                            \
  "mov (%%rsp), %%rax\n\t"                                                     \
  "xor %[mask], %%rax\n\t"                                                     \
  "mov %%rax, %c[rip](%%rdi)\n\t" ACCUMULATE_CHECKSUM()

#define STORE_CHECKSUM() "mov %%rdx, %c[__chksum](%%rdi)\n\t"
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
  // use registers to make sure values propagate correctly across the asm blocks
  [[maybe_unused]] register __UINTPTR_TYPE__ mask asm("rcx");
  [[maybe_unused]] register __UINT64_TYPE__ checksum asm("rdx");
  LOAD_CHKSUM_STATE_REGISTERS()
  asm volatile(
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
      :
#if LIBC_COPT_SETJMP_FORTIFICATION
      [checksum] "+r"(checksum)
#endif
      :
      [rbx] "i"(offsetof(__jmp_buf, rbx)), [rbp] "i"(offsetof(__jmp_buf, rbp)),
      [r12] "i"(offsetof(__jmp_buf, r12)), [r13] "i"(offsetof(__jmp_buf, r13)),
      [r14] "i"(offsetof(__jmp_buf, r14)), [r15] "i"(offsetof(__jmp_buf, r15)),
      [rsp] "i"(offsetof(__jmp_buf, rsp)),
      [rip] "i"(offsetof(__jmp_buf, rip))
#if LIBC_COPT_SETJMP_FORTIFICATION
      // clang-format off
      ,[rotation] "i"(jmpbuf::ROTATION)
      ,[__chksum] "i"(offsetof(__jmp_buf, __chksum))
      ,[mask] "r"(mask)
  // clang-format on
#endif
      : "rax");

  asm(R"(
    xorl %eax, %eax
    retq
  )");
}

} // namespace LIBC_NAMESPACE_DECL
