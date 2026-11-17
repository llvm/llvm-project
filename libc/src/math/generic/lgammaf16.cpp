//===-- Implementation of lgammaf16 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/lgammaf16.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/lgammaf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, lgammaf16, (float16 x)) {
  return math::lgammaf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
