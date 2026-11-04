//===-- Implementation of totalorderbf16 function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/totalorderbf16.h"
#include "src/__support/math/totalorderbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, totalorderbf16,
                   (const bfloat16 *x, const bfloat16 *y)) {
  return math::totalorderbf16(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
