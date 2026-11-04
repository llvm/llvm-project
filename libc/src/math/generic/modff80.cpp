//===-- Implementation of modff80 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/modff80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/modff80.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float80, modff80, (float80 x, float80 *iptr)) {
  Float80 iptr_f80 = cpp::bit_cast<Float80>(*iptr);
  Float80 res = math::modff80(cpp::bit_cast<Float80>(x), &iptr_f80);
  *iptr = cpp::bit_cast<float80>(iptr_f80);
  return cpp::bit_cast<float80>(res);
}

} // namespace LIBC_NAMESPACE_DECL
