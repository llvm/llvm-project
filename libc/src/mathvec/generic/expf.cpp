//===-- Single-precision SIMD e^x vector function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/mathvec/expf.h"
#include "src/__support/mathvec/expf.h"

namespace LIBC_NAMESPACE_DECL {

cpp::simd<float> expf(cpp::simd<float> x) { return mathvec::expf(x); }

} // namespace LIBC_NAMESPACE_DECL
