//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the canonicalizef80 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/canonicalizef80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/canonicalizef80.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float80;

LLVM_LIBC_FUNCTION(int, canonicalizef80, (float80 * cx, const float80 *x)) {
  Float80 cx_f80 = cpp::bit_cast<Float80>(*cx);
  Float80 x_f80 = cpp::bit_cast<Float80>(*x);
  int res = math::canonicalizef80(&cx_f80, &x_f80);
  *cx = cpp::bit_cast<float80>(cx_f80);
  return res;
}

} // namespace LIBC_NAMESPACE_DECL
