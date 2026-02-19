//===-- Implementation header for SIMD expf ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATHVEC_EXPF_H
#define LLVM_LIBC_SRC_MATHVEC_EXPF_H

#include "src/__support/CPP/simd.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

cpp::simd<float> expf(cpp::simd<float> x);

} // namespace LIBC_NAMESPACE_DECL

extern LIBC_NAMESPACE::cpp::simd<float> (*__llvm_libc_expf_cpp_simd_float)(
    LIBC_NAMESPACE::cpp::simd<float> x);

#endif // LLVM_LIBC_SRC_MATHVEC_EXPF_H
