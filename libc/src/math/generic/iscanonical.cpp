//===-- Implementation of iscanonical function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/iscanonical.h"
#include "src/__support/math/iscanonical.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, iscanonical, (double x)) {
  return math::iscanonical(x);
}

} // namespace LIBC_NAMESPACE_DECL
