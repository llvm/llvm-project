//===-- Implementation of longjmp (32-bit) --------------------------------===//
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

#if !defined(LIBC_TARGET_ARCH_IS_X86_32)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf, int)) {
  asm(R"(
      mov 0x4(%%esp), %%ecx
      mov 0x8(%%esp), %%eax
      cmpl $0x1, %%eax
      adcl $0x0, %%eax

      mov %c[ebx](%%ecx), %%ebx
      mov %c[esi](%%ecx), %%esi
      mov %c[edi](%%ecx), %%edi
      mov %c[ebp](%%ecx), %%ebp
      mov %c[esp](%%ecx), %%esp

      jmp *%c[eip](%%ecx)
      )" ::[ebx] "i"(offsetof(__jmp_buf, ebx)),
      [esi] "i"(offsetof(__jmp_buf, esi)), [edi] "i"(offsetof(__jmp_buf, edi)),
      [ebp] "i"(offsetof(__jmp_buf, ebp)), [esp] "i"(offsetof(__jmp_buf, esp)),
      [eip] "i"(offsetof(__jmp_buf, eip)));
}

} // namespace LIBC_NAMESPACE_DECL
