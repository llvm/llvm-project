//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Part of the Ripple vector library to support the HVX rotate instruction.
//
//==============================================================================

#include <hexagon_protos.h>
#include <ripple_hvx.h>

#include "HVX_Rotate.h"

#define _decl_rotate_impl(N, T, C_T, shft)                                     \
  RIPPLE_INTRIN_INLINE v##N##T ripple_pure_hvx_rotate_to_lower_##T(            \
      v##N##T in, int32_t n) {                                                 \
    int32_t shift = n << shft;                                                 \
    C_T dummy = 0;                                                             \
    return hvx_cast_from_i32(Q6_V_vror_VR(hvx_cast_to_i32(in), shift), dummy); \
  }

_decl_rotate_impl(128, i8, int8_t, 0);
_decl_rotate_impl(128, u8, uint8_t, 0);
_decl_rotate_impl(64, i16, int16_t, 1);
_decl_rotate_impl(64, u16, uint16_t, 1);
#if __has_bf16__
_decl_rotate_impl(64, bf16, __bf16, 1);
#endif
#if __has_Float16__
_decl_rotate_impl(64, f16, _Float16, 1);
#endif
_decl_rotate_impl(32, i32, int32_t, 2);
_decl_rotate_impl(32, u32, uint32_t, 2);
_decl_rotate_impl(32, f32, float, 2);
_decl_rotate_impl(16, i64, int64_t, 3);
_decl_rotate_impl(16, u64, uint64_t, 3);
_decl_rotate_impl(16, f64, double, 3);

#undef _decl_rotate
