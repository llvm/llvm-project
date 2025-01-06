//===-- Implementation of f16sqrtl function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16sqrtl.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16sqrtl, (long double x)) {
  return fputil::sqrt<float16>(x);
}

} // namespace LIBC_NAMESPACE_DECL
