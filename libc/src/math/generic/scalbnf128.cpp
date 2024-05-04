//===-- Implementation of scalbnf128 function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalbnf128.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float128, scalbnf128, (float128 x, int n)) {
// TODO: should be switched to use `FLT_RADIX` in hdr/float_macros.h" instead
// see: https://github.com/llvm/llvm-project/issues/90496
#if !defined(__FLT_RADIX__)
#error __FLT_RADIX__ undefined.
#elif __FLT_RADIX__ != 2
#error __FLT_RADIX__!=2, unimplemented.
#else
  return fputil::ldexp(x, n);
#endif
}

} // namespace LIBC_NAMESPACE
