//===-- Implementation of canonicalizef128 function------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/canonicalizef128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/canonicalizef128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(int, canonicalizef128, (float128 * cx, const float128 *x)) {
  Float128 cx_val{};
  const Float128 x_val = cpp::bit_cast<Float128>(*x);
  int result = math::canonicalizef128(&cx_val, &x_val);
  *cx = cpp::bit_cast<float128>(cx_val);
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
