//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the copysignf80 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/copysignf80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/copysignf80.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float80;

LLVM_LIBC_FUNCTION(float80, copysignf80, (float80 x, float80 y)) {
  return cpp::bit_cast<float80>(
      math::copysignf80(cpp::bit_cast<Float80>(x), cpp::bit_cast<Float80>(y)));
}

} // namespace LIBC_NAMESPACE_DECL
