//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the single-precision SIMD expm1 vector function.
///
//===----------------------------------------------------------------------===//

#include "src/mathvec/expm1f.h"
#include "src/__support/mathvec/expm1f.h"
#include "src/mathvec/abi_prefix.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpp::simd<float>, expm1f, (cpp::simd<float> x),
                   LIBC_VFABI_FLOAT_SYMBOL(expm1f)) {
  return mathvec::expm1f(x);
}

} // namespace LIBC_NAMESPACE_DECL
