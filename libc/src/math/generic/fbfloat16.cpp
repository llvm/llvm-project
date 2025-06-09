//===-- Implementation of fbfloat16 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fbfloat16.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fbfloat16, (LIBC_NAMESPACE::bfloat16 x)) {
  if (x.bits == 0)
    return 0.0f;
  else
    return 1.0f;
}

} // namespace LIBC_NAMESPACE_DECL
