//===-- Implementation of sigsetjmp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/sigsetjmp.h"
#include "hdr/offsetof_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"
#include "src/setjmp/sigsetjmp_epilogue.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid file include"
#endif
namespace LIBC_NAMESPACE_DECL {
#ifdef __i386__
[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, sigsetjmp, (sigjmp_buf buf)) {
  asm(R"(
      mov 8(%%esp), %%ecx
      jecxz .Lnosave

      mov 4(%%esp), %%eax
      pop %c[retaddr](%%eax)
      mov %%ebx, %c[extra](%%eax)
      mov %%eax, %%ebx
      call %P[setjmp]
      push %c[retaddr](%%ebx)
      mov %%ebx,4(%%esp)
      mov %%eax,8(%%esp)
      mov %c[extra](%%ebx), %%ebx
      jmp %P[epilogue]
      
.Lnosave:
      jmp %P[setjmp])" ::[retaddr] "i"(offsetof(__jmp_buf, sig_retaddr)),
      [extra] "i"(offsetof(__jmp_buf, sig_extra)), [setjmp] "X"(setjmp),
      [epilogue] "X"(sigsetjmp_epilogue)
      : "eax", "ebx", "ecx");
}
#else
[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, sigsetjmp, (sigjmp_buf, int)) {
  asm(R"(
      test %%esi, %%esi
      jz .Lnosave

      pop %c[retaddr](%%rdi)
      mov %%rbx, %c[extra](%%rdi)
      mov %%rdi, %%rbx
      call %P[setjmp]
      push %c[retaddr](%%rbx)
      mov %%rbx, %%rdi
      mov %%eax, %%esi
      mov %c[extra](%%rdi), %%rbx
      jmp %P[epilogue]
      
.Lnosave:
      jmp %P[setjmp])" ::[retaddr] "i"(offsetof(__jmp_buf, sig_retaddr)),
      [extra] "i"(offsetof(__jmp_buf, sig_extra)), [setjmp] "X"(setjmp),
      [epilogue] "X"(sigsetjmp_epilogue)
      : "rax", "rbx");
}
#endif

} // namespace LIBC_NAMESPACE_DECL
