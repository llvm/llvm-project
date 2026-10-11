//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of totalorderf128 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/totalorderf128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/totalorderf128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(int, totalorderf128,
                   (const float128 *x, const float128 *y)) {
  Float128 x_f128 = cpp::bit_cast<Float128>(*x);
  Float128 y_f128 = cpp::bit_cast<Float128>(*y);
  return math::totalorderf128(&x_f128, &y_f128);
}

} // namespace LIBC_NAMESPACE_DECL
