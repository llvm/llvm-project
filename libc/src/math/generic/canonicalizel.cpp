//===-- Implementation of canonicalizel function---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/canonicalizel.h"
#include "src/__support/math/canonicalizel.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, canonicalizel,
                   (long double *cx, const long double *x)) {
  return math::canonicalizel(cx, x);
}

} // namespace LIBC_NAMESPACE_DECL
