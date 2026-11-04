//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the totalorderf80 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/totalorderf80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/totalorderf80.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float80;

LLVM_LIBC_FUNCTION(int, totalorderf80,
                   (const float80 *x, const float80 *y)) {
  Float80 x_f80 = cpp::bit_cast<Float80>(*x);
  Float80 y_f80 = cpp::bit_cast<Float80>(*y);
  return math::totalorderf80(&x_f80, &y_f80);
}

} // namespace LIBC_NAMESPACE_DECL
