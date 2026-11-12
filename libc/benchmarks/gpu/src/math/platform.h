//===-- AMDGPU specific platform definitions for math support -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_MATH_AMDGPU_PLATFORM_H
#define LLVM_LIBC_SRC_MATH_AMDGPU_PLATFORM_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

#ifdef LIBC_TARGET_ARCH_IS_AMDGPU
// The ROCm device library uses control globals to alter codegen for the
// different targets. To avoid needing to link them in manually we simply
// define them here.
extern "C" {
extern const LIBC_INLINE_VAR uint8_t __oclc_unsafe_math_opt = 0;
extern const LIBC_INLINE_VAR uint8_t __oclc_daz_opt = 0;
extern const LIBC_INLINE_VAR uint8_t __oclc_correctly_rounded_sqrt32 = 1;
extern const LIBC_INLINE_VAR uint8_t __oclc_finite_only_opt = 0;
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9000;
}

// These aliases cause clang to emit the control constants with ODR linkage.
// This allows us to link against the symbols without preventing them from being
// optimized out or causing symbol collisions.
[[gnu::alias("__oclc_unsafe_math_opt")]] const uint8_t __oclc_unsafe_math_opt__;
[[gnu::alias("__oclc_daz_opt")]] const uint8_t __oclc_daz_opt__;
[[gnu::alias("__oclc_correctly_rounded_sqrt32")]] const uint8_t
    __oclc_correctly_rounded_sqrt32__;
[[gnu::alias("__oclc_finite_only_opt")]] const uint8_t __oclc_finite_only_opt__;
[[gnu::alias("__oclc_ISA_version")]] const uint32_t __oclc_ISA_version__;
#endif
} // namespace LIBC_NAMESPACE_DECL

// Forward declarations for the vendor math libraries.
extern "C" {
#ifdef AMDGPU_MATH_FOUND
double __ocml_atan2_f64(double, double);
float __ocml_atan2_f32(float, float);
double __ocml_exp_f64(double);
float __ocml_exp_f32(float);
float16 __ocml_exp_f16(float16);
double __ocml_log_f64(double);
float __ocml_log_f32(float);
float16 __ocml_log_f16(float16);
double __ocml_sin_f64(double);
float __ocml_sin_f32(float);
#endif

#ifdef NVPTX_MATH_FOUND
double __nv_atan2(double, double);
float __nv_atan2f(float, float);
double __nv_exp(double);
float __nv_expf(float);
double __nv_log(double);
float __nv_logf(float);
double __nv_sin(double);
float __nv_sinf(float);
#endif
}

#endif // LLVM_LIBC_SRC_MATH_AMDGPU_PLATFORM_H
