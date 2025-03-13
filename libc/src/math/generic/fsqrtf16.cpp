//===-- Implementation of fsqrtf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fsqrtf16.h"
#include "src/__support/FPUtil/generic/sqrt.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, fsqrtf16, (float16 x)) {
  return fputil::sqrt<float16>(x);
}

} // namespace LIBC_NAMESPACE_DECL
