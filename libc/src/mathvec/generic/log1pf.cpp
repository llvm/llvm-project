//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the single-precision SIMD log1p vector function.
///
//===----------------------------------------------------------------------===//

#include "src/mathvec/log1pf.h"
#include "src/__support/mathvec/log1pf.h"
#include "src/mathvec/abi_prefix.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpp::simd<float>, log1pf, (cpp::simd<float> x),
                   LIBC_VFABI_FLOAT_SYMBOL(log1pf)) {
  return mathvec::log1pf(x);
}

} // namespace LIBC_NAMESPACE_DECL
