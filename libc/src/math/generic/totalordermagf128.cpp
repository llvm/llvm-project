//===-- Implementation of totalordermagf128 function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/totalordermagf128.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, totalordermagf128,
                   (const float128 *x, const float128 *y)) {
  return static_cast<int>(fputil::totalordermag(*x, *y));
}

} // namespace LIBC_NAMESPACE_DECL
