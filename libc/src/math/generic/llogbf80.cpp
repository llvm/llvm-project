//===-- Implementation of llogbf80 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/llogbf80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/llogbf80.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, llogbf80, (float80 x)) {
  return math::llogbf80(cpp::bit_cast<Float80>(x));
}

} // namespace LIBC_NAMESPACE_DECL
