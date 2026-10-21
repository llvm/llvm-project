//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file
///  Double-precision tanh(x) function
///
//===----------------------------------------------------------------------===//

#include "src/math/tanh.h"
#include "src/__support/math/tanhf.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: Replace single-precision delegation stub with a full double-precision
// implementation once available.
LLVM_LIBC_FUNCTION(double, tanh, (double x)) {
  return static_cast<double>(math::tanhf(static_cast<float>(x)));
}

} // namespace LIBC_NAMESPACE_DECL
