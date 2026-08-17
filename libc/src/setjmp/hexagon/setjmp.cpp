//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the Hexagon implementation of `setjmp`.
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"

namespace LIBC_NAMESPACE_DECL {

// The Hexagon ABI callee-saved registers are r16-r27. Those are stored as
// register pairs, which requires the buffer to be 8-byte aligned. The stack
// pointer (r29), frame pointer (r30) and link register (r31) are stored
// individually; r28 is caller-saved so it does not need to be preserved.
[[gnu::naked]] LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  asm(R"(
      memd(r0+#0) = r17:16
      memd(r0+#8) = r19:18
      memd(r0+#16) = r21:20
      memd(r0+#24) = r23:22
      memd(r0+#32) = r25:24
      memd(r0+#40) = r27:26
      memw(r0+#48) = r29
      memw(r0+#52) = r30
      memw(r0+#56) = r31

      # Return zero.
      r0 = #0
      jumpr r31)");
}

} // namespace LIBC_NAMESPACE_DECL
