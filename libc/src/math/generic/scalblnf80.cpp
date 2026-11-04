//===-- Implementation of scalblnf80 function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalblnf80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/scalblnf80.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float80, scalblnf80, (float80 x, long n)) {
  return cpp::bit_cast<float80>(
      math::scalblnf80(cpp::bit_cast<Float80>(x), n));
}

} // namespace LIBC_NAMESPACE_DECL
