//===-- Implementation of faddbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/faddbf16.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// NOTE: this should be removed in lieu of operator overloads
LLVM_LIBC_FUNCTION(bfloat16, faddbf16, (bfloat16 x, bfloat16 y)) {
  fputil::DyadicFloat<16> xd(x);
  fputil::DyadicFloat<16> yd(y);
  fputil::DyadicFloat<16> zd = fputil::quick_add(xd, yd);
  return zd.as<bfloat16, /*ShouldSignalExceptions=*/true>();
}

} // namespace LIBC_NAMESPACE_DECL
