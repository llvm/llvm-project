//===-- Implementation of iscanonical function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/iscanonical.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#undef iscanonical
LLVM_LIBC_FUNCTION(int, iscanonical, (double x)) {
  double temp;
  return fputil::canonicalize(temp, x) == 0;
}

} // namespace LIBC_NAMESPACE_DECL
