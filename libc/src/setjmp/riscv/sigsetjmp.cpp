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

#if __riscv_xlen == 64
#define STORE(A, B, C) "sd " #A ", %c[" #B "](" #C ")\n\t"
#define LOAD(A, B, C) "ld " #A ", %c[" #B "](" #C ")\n\t"
#elif __riscv_xlen == 32
#define STORE(A, B, C) "sw " #A ", %c[" #B "](" #C ")\n\t"
#define LOAD(A, B, C) "lw " #A ", %c[" #B "](" #C ")\n\t"
#else
#error "Unsupported RISC-V architecture"
#endif

namespace LIBC_NAMESPACE_DECL {
[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, sigsetjmp, (sigjmp_buf, int)) {
  // clang-format off
  asm("beqz a1, .Lnosave\n\t"
      STORE(ra, retaddr, a0)
      STORE(s0, extra, a0)
      "mv s0, a0\n\t"
      "call %c[setjmp]\n\t"
      "mv a1, a0\n\t"
      "mv a0, s0\n\t"
      LOAD(s0, extra, a0)
      LOAD(ra, retaddr, a0)
      "tail %c[epilogue]\n"
".Lnosave:\n\t"
      "tail %c[setjmp]"
      // clang-format on
      ::[retaddr] "i"(offsetof(__jmp_buf, sig_retaddr)),
      [extra] "i"(offsetof(__jmp_buf, sig_extra)), [setjmp] "i"(setjmp),
      [epilogue] "i"(sigsetjmp_epilogue)
      : "a0", "a1", "s0");
}

} // namespace LIBC_NAMESPACE_DECL
