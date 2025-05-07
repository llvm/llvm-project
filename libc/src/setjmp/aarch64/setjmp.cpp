//===-- Implementation of setjmp for AArch64 ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]] LLVM_LIBC_FUNCTION(int, setjmp, ([[maybe_unused]] jmp_buf buf)) {
  // If BTI branch protection is in use, the compiler will automatically insert
  // a BTI here, so we don't need to make any extra effort to do so.

  asm(
#if __ARM_FEATURE_PAC_DEFAULT & 1
      // Sign the return address using the PAC A key.
      R"(
        paciasp
      )"
#elif __ARM_FEATURE_PAC_DEFAULT & 2
      // Sign the return address using the PAC B key.
      R"(
        pacibsp
      )"
#endif

      // Store all the callee-saved GPRs, including fp (x29) and also lr (x30).
      // Of course lr isn't normally callee-saved (the call instruction itself
      // can't help clobbering it), but we certainly need to save it for this
      // purpose.
      R"(
        stp x19, x20, [x0,  #0*16]
        stp x21, x22, [x0,  #1*16]
        stp x23, x24, [x0,  #2*16]
        stp x25, x26, [x0,  #3*16]
        stp x27, x28, [x0,  #4*16]
        stp x29, x30, [x0,  #5*16]
      )"

#if LIBC_COPT_SETJMP_AARCH64_RESTORE_PLATFORM_REGISTER
      // Store the stack pointer, and the platform register x18.
      R"(
        add x1, sp, #0
        stp x1, x18,  [x0,  #6*16]
      )"
#else
      // Store just the stack pointer.
      R"(
        add x1, sp, #0
        str x1,       [x0,  #6*16]
      )"
#endif

#if __ARM_FP
      // Store the callee-saved FP registers. AAPCS64 only requires the low 64
      // bits of v8-v15 to be preserved, i.e. each of d8,...,d15.
      R"(
        stp d8,  d9,  [x0,  #7*16]
        stp d10, d11, [x0,  #8*16]
        stp d12, d13, [x0,  #9*16]
        stp d14, d15, [x0, #10*16]
      )"
#endif

      // Set up return value of zero.
      R"(
        mov x0, #0
      )"

#if (__ARM_FEATURE_PAC_DEFAULT & 7) == 5
      // Authenticate the return address using the PAC A key, since the
      // compilation options ask for PAC protection even on leaf functions.
      R"(
        autiasp
      )"
#elif (__ARM_FEATURE_PAC_DEFAULT & 7) == 6
      // Same, but using the PAC B key.
      R"(
        autibsp
      )"
#endif

      R"(
        ret
      )");
}

} // namespace LIBC_NAMESPACE_DECL
