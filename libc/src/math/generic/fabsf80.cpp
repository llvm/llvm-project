//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the fabsf80 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/fabsf80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/fabsf80.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float80;

LLVM_LIBC_FUNCTION(float80, fabsf80, (float80 x)) {
  return cpp::bit_cast<float80>(math::fabsf80(cpp::bit_cast<Float80>(x)));
}

} // namespace LIBC_NAMESPACE_DECL
