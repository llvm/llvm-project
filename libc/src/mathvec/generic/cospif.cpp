//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the single-precision SIMD cospi vector function.
///
//===----------------------------------------------------------------------===//

#include "src/mathvec/cospif.h"
#include "src/__support/mathvec/cospif.h"
#include "src/mathvec/abi_prefix.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpp::simd<float>, cospif, (cpp::simd<float> x),
                   LIBC_VFABI_FLOAT_SYMBOL(cospif)) {
  return mathvec::cospif(x);
}

} // namespace LIBC_NAMESPACE_DECL
