//===----- hlsl_intrinsics.h - HLSL definitions for intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_INTRINSICS_H_
#define _HLSL_HLSL_INTRINSICS_H_

__attribute__((clang_builtin_alias(__builtin_hlsl_wave_active_count_bits))) uint
WaveActiveCountBits(bool bBit);

__attribute__((clang_builtin_alias(__builtin_abs))) int abs(int In);
__attribute__((clang_builtin_alias(__builtin_labs))) int64_t abs(int64_t In);
__attribute__((clang_builtin_alias(__builtin_fabsf))) float abs(float In);
__attribute__((clang_builtin_alias(__builtin_fabs))) double abs(double In);

#ifdef __HLSL_ENABLE_16_BIT
__attribute__((clang_builtin_alias(__builtin_fabsf16))) half abs(half In);
#endif

#endif //_HLSL_HLSL_INTRINSICS_H_
