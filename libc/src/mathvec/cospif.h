//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for single-precision SIMD cospi.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATHVEC_COSPIF_H
#define LLVM_LIBC_SRC_MATHVEC_COSPIF_H

#include "src/__support/CPP/simd.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

cpp::simd<float> cospif(cpp::simd<float> x);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATHVEC_COSPIF_H
