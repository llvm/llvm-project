//===-- Implementation of setpayloadsigf16 function -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadsigf16.h"
#include "src/__support/math/setpayloadsigf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpayloadsigf16, (float16 * res, float16 pl)) {
  return math::setpayloadsigf16(res, pl);
}

} // namespace LIBC_NAMESPACE_DECL
