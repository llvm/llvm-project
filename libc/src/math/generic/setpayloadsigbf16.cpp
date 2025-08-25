//===-- Implementation of setpayloadsigbf16 function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadsigbf16.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpayloadsigbf16, (bfloat16 * res, bfloat16 pl)) {
  return static_cast<int>(fputil::setpayload</*IsSignaling=*/true>(*res, pl));
}

} // namespace LIBC_NAMESPACE_DECL
