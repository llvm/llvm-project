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
#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid file include"
#endif
namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, sigsetjmp, (sigjmp_buf buf, int save_mask)) {
  asm(R"(
      test %%esi, %%esi
      jz .Lnosave

      pop %c[retaddr](%%rdi)         // pop the return address into the buffer
      mov %%rbx, %c[extra](%%rdi)   // move the value in %rbx to the 'extra' field in the buffer
      mov %%rdi, %%rbx              // move the buffer address to %rbx
      call %P[setjmp]               // call setjmp
      push %c[retaddr](%%rbx)       // push return address back into buffer
      mov %%rbx, %%rdi              // move buffer address back into %rdi
      mov %%eax, %%esi              // move setjmp return value to %esi (for save_mask)
      mov %c[extra](%%rdi), %%rbx   // restore the extra field
      jmp %P[epilogue]              // jump to epilogue

.Lnosave:
      jmp %P[setjmp]                // jump directly to setjmp if no save mask is provided
  )" :: [retaddr] "i"(offsetof(__jmp_buf, sig_retaddr)),
      [extra] "i"(offsetof(__jmp_buf, sig_extra)), [setjmp] "X"(setjmp),
      [epilogue] "X"(sigsetjmp_epilogue)
      : "rax", "rbx", "rdi", "rsi");
}

} // namespace LIBC_NAMESPACE_DECL