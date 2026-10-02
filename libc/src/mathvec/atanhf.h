//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for single-precision SIMD atanh.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATHVEC_ATANHF_H
#define LLVM_LIBC_SRC_MATHVEC_ATANHF_H

#include "src/__support/CPP/simd.h"

namespace LIBC_NAMESPACE_DECL {

cpp::simd<float> atanhf(cpp::simd<float> x);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATHVEC_ATANHF_H
