//===-- Implementation of ceilf128 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ceilf128.h"
#include "src/__support/math/ceilf128.h"
#include "src/__support/FPUtil/float128.h"

using LIBC_NAMESPACE::fputil::Float128;

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(Float128, ceilf128, (Float128 x)) {
  return math::ceilf128(x);
}

} // namespace LIBC_NAMESPACE_DECL
