//===-- Implementation of longjmp for AArch64 -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: if MTE stack tagging is in use (-fsanitize=memtag-stack), we need to
// iterate over the region between the old and new values of sp, using STG or
// ST2G instructions to clear the memory tags on the invalidated region of the
// stack. But this requires a means of finding out that we're in that mode, and
// as far as I can see there isn't currently a predefined macro for that.
//
// (__ARM_FEATURE_MEMORY_TAGGING only indicates whether the target architecture
// supports the MTE instructions, not whether the compiler is configured to use
// them.)

[[gnu::naked]] LLVM_LIBC_FUNCTION(void, longjmp,
                                  ([[maybe_unused]] jmp_buf buf,
                                   [[maybe_unused]] int val)) {
  // If BTI branch protection is in use, the compiler will automatically insert
  // a BTI here, so we don't need to make any extra effort to do so.

  // If PAC branch protection is in use, there's no need to sign the return
  // address at the start of longjmp, because we're not going to use it anyway!

  asm(
      // Reload the callee-saved GPRs, including fp and lr.
      R"(
        ldp x19, x20, [x0,  #0*16]
        ldp x21, x22, [x0,  #1*16]
        ldp x23, x24, [x0,  #2*16]
        ldp x25, x26, [x0,  #3*16]
        ldp x27, x28, [x0,  #4*16]
        ldp x29, x30, [x0,  #5*16]
      )"

#if LIBC_COPT_SETJMP_AARCH64_RESTORE_PLATFORM_REGISTER
      // Reload the stack pointer, and the platform register x18.
      R"(
        ldp x2,  x18, [x0,  #6*16]
        mov sp, x2
      )"
#else
      // Reload just the stack pointer.
      R"(
        ldr x2,       [x0,  #6*16]
        mov sp, x2
      )"
#endif

#if __ARM_FP
      // Reload the callee-saved FP registers.
      R"(
        ldp d8,  d9,  [x0,  #7*16]
        ldp d10, d11, [x0,  #8*16]
        ldp d12, d13, [x0,  #9*16]
        ldp d14, d15, [x0, #10*16]
      )"
#endif

      // Calculate the return value.
      R"(
        cmp w1, #0
        cinc w0, w1, eq
      )"

#if __ARM_FEATURE_PAC_DEFAULT & 1
      // Authenticate the return address using the PAC A key.
      R"(
        autiasp
      )"
#elif __ARM_FEATURE_PAC_DEFAULT & 2
      // Authenticate the return address using the PAC B key.
      R"(
        autibsp
      )"
#endif

      R"(
        ret
      )");
}

} // namespace LIBC_NAMESPACE_DECL
