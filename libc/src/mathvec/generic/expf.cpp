//===-- Single-precision SIMD e^x vector function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/mathvec/expf.h"
#include "src/__support/mathvec/expf.h"
#include <typeinfo>

namespace LIBC_NAMESPACE_DECL {

cpp::simd<float> expf(cpp::simd<float> x) { return mathvec::expf(x); }

#ifdef __clang__
#define EXPF_MANGLED_NAME _ZN22__llvm_libc_22_0_0_git5expfEDv8_f
#else
#ifdef __some_other_compiler__
#define EXPF_MANGLED_NAME other_mangled_name
#endif
#endif
extern expf_simd_float_ftype EXPF_MANGLED_NAME;

} // namespace LIBC_NAMESPACE_DECL

extern const LIBC_NAMESPACE::expf_simd_float_ftype __expf_simd_float =
    reinterpret_cast<LIBC_NAMESPACE::expf_simd_float_ftype>(
        &LIBC_NAMESPACE::EXPF_MANGLED_NAME);
