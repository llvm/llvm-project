//===-- Implementation of ufromfpxf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ufromfpxf16.h"
#include "src/__support/math/ufromfpxf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, ufromfpxf16,
                   (float16 x, int rnd, unsigned int width)) {
  return math::ufromfpxf16(x, rnd, width);
}

} // namespace LIBC_NAMESPACE_DECL
