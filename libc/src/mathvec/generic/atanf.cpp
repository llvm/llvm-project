//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the single-precision SIMD atan vector function.
///
//===----------------------------------------------------------------------===//

#include "src/mathvec/atanf.h"
#include "src/__support/mathvec/atanf.h"
#include "src/mathvec/abi_prefix.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpp::simd<float>, atanf, (cpp::simd<float> x),
                   LIBC_VFABI_FLOAT_SYMBOL(atanf)) {
  return mathvec::atanf(x);
}

} // namespace LIBC_NAMESPACE_DECL
