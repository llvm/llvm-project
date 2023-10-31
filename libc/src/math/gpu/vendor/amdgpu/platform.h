//===-- AMDGPU specific platform definitions for math support -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GPU_AMDGPU_PLATFORM_H
#define LLVM_LIBC_SRC_MATH_GPU_AMDGPU_PLATFORM_H

#include "src/__support/macros/attributes.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

// The ROCm device library uses control globals to alter codegen for the
// different targets. To avoid needing to link them in manually we simply
// define them here.
extern "C" {

// Disable unsafe math optimizations in the implementation.
extern const LIBC_INLINE_VAR uint8_t __oclc_unsafe_math_opt = 0;

// Disable denormalization at zero optimizations in the implementation.
extern const LIBC_INLINE_VAR uint8_t __oclc_daz_opt = 0;

// Disable rounding optimizations for 32-bit square roots.
extern const LIBC_INLINE_VAR uint8_t __oclc_correctly_rounded_sqrt32 = 1;

// Disable finite math optimizations.
extern const LIBC_INLINE_VAR uint8_t __oclc_finite_only_opt = 0;

#if defined(__gfx700__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 7000;
#elif defined(__gfx701__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 7001;
#elif defined(__gfx702__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 7002;
#elif defined(__gfx703__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 7003;
#elif defined(__gfx704__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 7004;
#elif defined(__gfx705__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 7005;
#elif defined(__gfx801__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 8001;
#elif defined(__gfx802__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 8002;
#elif defined(__gfx803__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 8003;
#elif defined(__gfx805__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 8005;
#elif defined(__gfx810__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 8100;
#elif defined(__gfx900__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9000;
#elif defined(__gfx902__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9002;
#elif defined(__gfx904__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9004;
#elif defined(__gfx906__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9006;
#elif defined(__gfx908__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9008;
#elif defined(__gfx909__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9009;
#elif defined(__gfx90a__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9010;
#elif defined(__gfx90c__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9012;
#elif defined(__gfx940__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 9400;
#elif defined(__gfx1010__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10100;
#elif defined(__gfx1011__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10101;
#elif defined(__gfx1012__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10102;
#elif defined(__gfx1013__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10103;
#elif defined(__gfx1030__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10300;
#elif defined(__gfx1031__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10301;
#elif defined(__gfx1032__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10302;
#elif defined(__gfx1033__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10303;
#elif defined(__gfx1034__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10304;
#elif defined(__gfx1035__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10305;
#elif defined(__gfx1036__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 10306;
#elif defined(__gfx1100__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 11000;
#elif defined(__gfx1101__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 11001;
#elif defined(__gfx1102__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 11002;
#elif defined(__gfx1103__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 11003;
#elif defined(__gfx1150__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 11500;
#elif defined(__gfx1151__)
extern const LIBC_INLINE_VAR uint32_t __oclc_ISA_version = 11501;
#else
#error "Unknown AMDGPU architecture"
#endif
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

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_GPU_AMDGPU_PLATFORM_H
