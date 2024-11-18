//===-- Implementation of setjmp (32-bit) ---------------------------------===//
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

#if !defined(LIBC_TARGET_ARCH_IS_X86_32)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {
[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  asm(R"(
      mov 4(%%esp), %%eax

      mov %%ebx, %c[ebx](%%eax)
      mov %%esi, %c[esi](%%eax)
      mov %%edi, %c[edi](%%eax)
      mov %%ebp, %c[ebp](%%eax)

      lea 4(%%esp), %%ecx
      mov %%ecx, %c[esp](%%eax)

      mov (%%esp), %%ecx
      mov %%ecx, %c[eip](%%eax)

      xorl %%eax, %%eax
      retl)" ::[ebx] "i"(offsetof(__jmp_buf, ebx)),
      [esi] "i"(offsetof(__jmp_buf, esi)), [edi] "i"(offsetof(__jmp_buf, edi)),
      [ebp] "i"(offsetof(__jmp_buf, ebp)), [esp] "i"(offsetof(__jmp_buf, esp)),
      [eip] "i"(offsetof(__jmp_buf, eip))
      : "eax", "ecx");
}

} // namespace LIBC_NAMESPACE_DECL
