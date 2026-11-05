//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the single-precision SIMD sinpi vector function.
///
//===----------------------------------------------------------------------===//

#include "src/mathvec/sinpif.h"
#include "src/__support/mathvec/sinpif.h"
#include "src/mathvec/abi_prefix.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpp::simd<float>, sinpif, (cpp::simd<float> x),
                   LIBC_VFABI_FLOAT_SYMBOL(sinpif)) {
  return mathvec::sinpif(x);
}

} // namespace LIBC_NAMESPACE_DECL
