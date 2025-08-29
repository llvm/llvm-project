//===----------------------------- HVX_Bfloat.h ----------------------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===
//           Header to access the bfloat conversion API from Ripple.
//===------------------------------------------------------------------------===

#pragma once

#include "vector_types.h"
#include <ripple.h>

#if __has_bf16__

#ifdef __cplusplus
extern "C" {
#else
extern {
#endif

/// \brief Bare truncation from fp32 to bf16:
///  - assumes that the elements of `in` are non-NaNs.
///  - Gets a sub-optimal rounding error from just truncating
v64bf16 ripple_ew_pure_to_bf16_trunc(v64f32 in);

/// \brief fp32 -> bf16 truncation that's aware of NaNs.
/// Use if there is a possibility that `in`
/// contains NaNs, but if you don't care about the rounding error introduced
/// by a bare truncation.
v64bf16 ripple_ew_pure_to_bf16_nan(v64f32 vin);

/// \brief most accurate fp32 --> bf16 conversion.
/// Uses even rounding and is aware of NaN conversions.
v64bf16 ripple_ew_pure_to_bf16_round(v64f32 vin);
}

#endif // __has_bf16__
