//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the Hexagon implementation of `longjmp`.
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Restore the registers saved by setjmp, see the comment there for the layout
// of the buffer.
[[gnu::naked]] LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf buf, int val)) {
  asm(R"(
      r17:16 = memd(r0+#0)
      r19:18 = memd(r0+#8)
      r21:20 = memd(r0+#16)
      r23:22 = memd(r0+#24)
      r25:24 = memd(r0+#32)
      r27:26 = memd(r0+#40)
      r29 = memw(r0+#48)
      r30 = memw(r0+#52)
      r31 = memw(r0+#56)

      # return val ?: 1;
      { p0 = cmp.eq(r1,#0); if (p0.new) r1 = #1 }
      r0 = r1
      jumpr r31)");
}

} // namespace LIBC_NAMESPACE_DECL
