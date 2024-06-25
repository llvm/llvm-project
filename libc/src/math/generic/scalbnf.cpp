//===-- Single-precision scalbnf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalbnf.h"
#include "hdr/float_macros.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"

#if FLT_RADIX != 2
#error "FLT_RADIX != 2 is not supported."
#endif

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, scalbnf, (float x, int n)) {
  return fputil::ldexp(x, n);
}

} // namespace LIBC_NAMESPACE
