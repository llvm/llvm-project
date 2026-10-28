//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file
///  Double-precision sinh(x) function
///
//===----------------------------------------------------------------------===//

#include "src/math/sinh.h"
#include "src/__support/math/sinhf.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: Replace single-precision delegation stub with a full double-precision
// implementation once available.
LLVM_LIBC_FUNCTION(double, sinh, (double x)) {
  return static_cast<double>(math::sinhf(static_cast<float>(x)));
}

} // namespace LIBC_NAMESPACE_DECL
