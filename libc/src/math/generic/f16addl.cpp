//===-- Implementation of f16addl function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16addl.h"
#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16addl, (long double x, long double y)) {
  return fputil::generic::add<float16>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
