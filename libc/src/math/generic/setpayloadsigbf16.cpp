//===-- Implementation of setpayloadsigbf16 function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadsigbf16.h"
#include "src/__support/math/setpayloadsigbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpayloadsigbf16, (bfloat16 * res, bfloat16 pl)) {
  return math::setpayloadsigbf16(res, pl);
}

} // namespace LIBC_NAMESPACE_DECL
