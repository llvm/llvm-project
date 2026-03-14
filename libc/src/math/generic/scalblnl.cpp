//===-- Implementation of scalblnl function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalblnl.h"
#include "hdr/float_macros.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#if FLT_RADIX != 2
#error "FLT_RADIX != 2 is not supported."
#endif

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, scalblnl, (long double x, long n)) {
  return fputil::ldexp(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
