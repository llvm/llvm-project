//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the single-precision SIMD erf vector function.
///
//===----------------------------------------------------------------------===//

#include "src/mathvec/erff.h"
#include "src/__support/mathvec/erff.h"
#include "src/mathvec/abi_prefix.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpp::simd<float>, erff, (cpp::simd<float> x),
                   LIBC_VFABI_FLOAT_SYMBOL(erff)) {
  return mathvec::erff(x);
}

} // namespace LIBC_NAMESPACE_DECL
