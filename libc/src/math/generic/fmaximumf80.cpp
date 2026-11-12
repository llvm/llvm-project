//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the fmaximumf80 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/fmaximumf80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/float80.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/fmaximumf80.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float80;

LLVM_LIBC_FUNCTION(float80, fmaximumf80, (float80 x, float80 y)) {
  return cpp::bit_cast<float80>(
      math::fmaximumf80(cpp::bit_cast<Float80>(x), cpp::bit_cast<Float80>(y)));
}

} // namespace LIBC_NAMESPACE_DECL
