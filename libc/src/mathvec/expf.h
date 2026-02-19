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

// What plain C-type corresponds to cpp::simd<float>? Replace uint64_t with
// that, then uncomment the static assert. We have to trust the callers to
// build the type properly, which is type unsafe.
using cpp_simd_float = uint64_t;
// static_assert(sizeof(cpp_simd_float) == sizeof(cpp::simd<float>));
static_assert(cpp::is_trivially_copyable<cpp::simd<float>>::value == true);
static_assert(cpp::is_trivially_copyable<cpp_simd_float>::value == true);
using expf_simd_float_ftype = cpp_simd_float (*)(cpp_simd_float);
} // namespace LIBC_NAMESPACE_DECL

extern const LIBC_NAMESPACE::expf_simd_float_ftype __expf_cpp_simd_float;

#endif // LLVM_LIBC_SRC_MATHVEC_EXPF_H
