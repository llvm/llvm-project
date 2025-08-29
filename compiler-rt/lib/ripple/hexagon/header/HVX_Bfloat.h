//===-------------------------- HVX_Bfloat.h ------------------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===
//       Internal header to access the bfloat conversion API from Ripple.
//===------------------------------------------------------------------------===

#include "ripple.h"

#if __has_bf16__

#ifdef __cplusplus
extern "C" {
#else
extern {
#endif

/// \brief Bare truncation from fp32 to bf16:
///  - assumes that the elements of `in` are non-NaNs.
///  - Gets a sub-optimal rounding error from just truncating
__bf16 to_bf16_trunc(float in);

/// \brief fp32 -> bf16 truncation that's aware of NaNs.
/// Use if there is a possibility that `in`
/// contains NaNs, but if you don't care about the rounding error introduced
/// by a bare truncation.
__bf16 to_bf16_nan(float in);

/// \brief most accurate fp32 --> bf16 conversion.
/// Uses even rounding and is aware of NaN conversions.
__bf16 to_bf16_round(float vin);
}

#endif // __has_bf16__
