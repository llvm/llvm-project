//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// HVX scalar-broadcast multiply (vmpy) and multiply-accumulate (vmpyacc)
// runtime library.
//
// Each function is a thin wrapper around a Hexagon HVX built-in intrinsic.
// RIPPLE_INTRIN_INLINE ensures the function is always inlined and emitted as
// a Ripple pure HVX intrinsic, allowing the Ripple compiler pass to recognize
// and replace calls with the corresponding HVX vector instruction.
//
// Naming convention:
//   ripple_ew_pure_hvx_<outType>_<op>_<inType><rType>
//   where <op> is one of:
//     vmpy        - multiply vector by scalar:       out[i] = v[i] * r
//     vmpyacc     - multiply-accumulate:             out[i] = x[i] + v[i] * r
//     vmpyacc_sat - multiply-accumulate w/ saturate: out[i] = sat(x[i] + v[i] *
//     r)

#include "hexagon_types.h"
#include "lib_func_attrib.h"
#pragma once
#include <ripple_hvx.h>
#ifdef __cplusplus
extern "C" {
#endif

//
// _____________________ vmpy ____________________________
// Multiply each vector element by a scalar, widening the result.
//
// OUTPUT ORDER NOTE: All widening vmpy/vmpyacc intrinsics (Q6_W*_vmpy_*)
// produce results in ODD/EVEN INTERLEAVED order:
//   lo(result)[i] = input[2*i]   (even-indexed elements)
//   hi(result)[i] = input[2*i+1] (odd-indexed elements)
// This is the natural HW output.  The _noshuff variants below apply
// Q6_W_vdeal_VVR to restore sequential (original input) order.
//

// out[i] = (int16_t)((uint8_t)v[i] * (int8_t)r)
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
RIPPLE_INTRIN_INLINE v128i16 ripple_ew_pure_hvx_i16_vmpy_u8i8(v128u8 v,
                                                              signed char r) {
  return Q6_Wh_vmpy_VubRb(v, r);
}

// out[i] = (uint16_t)((uint8_t)v[i] * (uint8_t)r)
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
RIPPLE_INTRIN_INLINE v128u16 ripple_ew_pure_hvx_u16_vmpy_u8u8(v128u8 v,
                                                              unsigned char r) {
  return Q6_Wuh_vmpy_VubRub(v, r);
}

// out[i] = (uint32_t)((uint16_t)v[i] * (uint16_t)r)
// Note: r encodes two u16 values packed into a u32 (hence _2u16 suffix).
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
RIPPLE_INTRIN_INLINE v64u32
ripple_ew_pure_hvx_u32_vmpy_u16_2u16(v64u16 v, unsigned int r) {
  return Q6_Wuw_vmpy_VuhRuh(v, r);
}

// out[i] = (int32_t)((int16_t)v[i] * (int16_t)r)
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
RIPPLE_INTRIN_INLINE v64i32 ripple_ew_pure_hvx_i32_vmpy_i16i16(v64i16 v,
                                                               short r) {
  return Q6_Ww_vmpy_VhRh(v, r);
}

// out[i] = sat16((int16_t)v[i] * (int16_t)r << 1)
// OUTPUT: sequential -- same element count as input, no reordering
RIPPLE_INTRIN_INLINE v64i16 ripple_ew_pure_hvx_i16_vmpy_i16i16_s1_sat(v64i16 v,
                                                                      short r) {
  return Q6_Vh_vmpy_VhRh_s1_sat(v, r);
}

// out[i] = sat16(((int16_t)v[i] * (int16_t)r << 1) + 0x8000 >> 16)
// OUTPUT: sequential -- same element count as input, no reordering
RIPPLE_INTRIN_INLINE v64i16
ripple_ew_pure_hvx_i16_vmpy_i16i16_s1_rnd_sat(v64i16 v, short r) {
  return Q6_Vh_vmpy_VhRh_s1_rnd_sat(v, r);
}

//
// _____________________ vmpy _noshuff ____________________________
// Same operation as vmpy above, but output is deinterleaved back to
// sequential (original input) order via Q6_W_vdeal_VVR.
// Use these when the caller expects out[i] to correspond to input[i].
//

// out[i] = (int16_t)((uint8_t)v[i] * (int8_t)r)
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v128i16
ripple_ew_pure_hvx_i16_vmpy_u8i8_noshuff(v128u8 v, signed char r) {
  v128i16 interleaved = Q6_Wh_vmpy_VubRb(v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 2);
}

// out[i] = (uint16_t)((uint8_t)v[i] * (uint8_t)r)
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v128u16
ripple_ew_pure_hvx_u16_vmpy_u8u8_noshuff(v128u8 v, unsigned char r) {
  v128u16 interleaved = Q6_Wuh_vmpy_VubRub(v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 2);
}

// out[i] = (uint32_t)((uint16_t)v[i] * (uint16_t)r)
// Note: r encodes two u16 values packed into a u32 (hence _2u16 suffix).
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v64u32
ripple_ew_pure_hvx_u32_vmpy_u16_2u16_noshuff(v64u16 v, unsigned int r) {
  v64u32 interleaved = Q6_Wuw_vmpy_VuhRuh(v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 4);
}

// out[i] = (int32_t)((int16_t)v[i] * (int16_t)r)
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v64i32
ripple_ew_pure_hvx_i32_vmpy_i16i16_noshuff(v64i16 v, short r) {
  v64i32 interleaved = Q6_Ww_vmpy_VhRh(v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 4);
}

//
// _____________________ vmpyacc ____________________________
// Multiply-accumulate: add scalar-broadcast product into an accumulator vector.
//

// out[i] = x[i] + (int16_t)((uint8_t)v[i] * (int8_t)r)
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
// NOTE: accumulator x must also be in odd/even interleaved order.
RIPPLE_INTRIN_INLINE v128i16
ripple_ew_pure_hvx_i16_vmpyacc_i16u8i8(v128i16 x, v128u8 v, signed char r) {
  return Q6_Wh_vmpyacc_WhVubRb(x, v, r);
}

// out[i] = x[i] + (uint16_t)((uint8_t)v[i] * (uint8_t)r)
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
// NOTE: accumulator x must also be in odd/even interleaved order.
RIPPLE_INTRIN_INLINE v128u16
ripple_ew_pure_hvx_u16_vmpyacc_u16u8u8(v128u16 x, v128u8 v, unsigned char r) {
  return Q6_Wuh_vmpyacc_WuhVubRub(x, v, r);
}

// out[i] = x[i] + (uint32_t)((uint16_t)v[i] * (uint16_t)r)
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
// NOTE: accumulator x must also be in odd/even interleaved order.
RIPPLE_INTRIN_INLINE v64u32
ripple_ew_pure_hvx_u32_vmpyacc_u32u16u16(v64u32 x, v64u16 v, unsigned short r) {
  return Q6_Wuw_vmpyacc_WuwVuhRuh(x, v, r);
}

// out[i] = x[i] + (int32_t)((int16_t)v[i] * (int16_t)r)
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
// NOTE: accumulator x must also be in odd/even interleaved order.
RIPPLE_INTRIN_INLINE v64i32 ripple_ew_pure_hvx_i32_vmpyacc_i32i16i16(v64i32 x,
                                                                     v64i16 v,
                                                                     short r) {
  return Q6_Ww_vmpyacc_WwVhRh(x, v, r);
}

// out[i] = sat32(x[i] + (int32_t)((int16_t)v[i] * (int16_t)r))
// OUTPUT: odd/even interleaved -- lo=even elements, hi=odd elements
// NOTE: accumulator x must also be in odd/even interleaved order.
RIPPLE_INTRIN_INLINE v64i32
ripple_ew_pure_hvx_i32_vmpyacc_i32i16i16_sat(v64i32 x, v64i16 v, short r) {
  return Q6_Ww_vmpyacc_WwVhRh_sat(x, v, r);
}

//
// _____________________ vmpyacc _noshuff ____________________________
// Same as vmpyacc above, but both input accumulator and output are in
// sequential (original input) order.
// The accumulator x is in sequential order, so it must be shuffled into
// interleaved order before passing to the HVX intrinsic (which requires an
// interleaved accumulator), and the result is then vdeal'd back to sequential.
//

// out[i] = x[i] + (int16_t)((uint8_t)v[i] * (int8_t)r)
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v128i16 ripple_ew_pure_hvx_i16_vmpyacc_i16u8i8_noshuff(
    v128i16 x, v128u8 v, signed char r) {
  v128i16 x_interleaved = Q6_W_vshuff_VVR(Q6_V_hi_W(x), Q6_V_lo_W(x), 2);
  v128i16 interleaved = Q6_Wh_vmpyacc_WhVubRb(x_interleaved, v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 2);
}

// out[i] = x[i] + (uint16_t)((uint8_t)v[i] * (uint8_t)r)
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v128u16 ripple_ew_pure_hvx_u16_vmpyacc_u16u8u8_noshuff(
    v128u16 x, v128u8 v, unsigned char r) {
  v128u16 x_interleaved = Q6_W_vshuff_VVR(Q6_V_hi_W(x), Q6_V_lo_W(x), 2);
  v128u16 interleaved = Q6_Wuh_vmpyacc_WuhVubRub(x_interleaved, v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 2);
}

// out[i] = x[i] + (uint32_t)((uint16_t)v[i] * (uint16_t)r)
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v64u32 ripple_ew_pure_hvx_u32_vmpyacc_u32u16u16_noshuff(
    v64u32 x, v64u16 v, unsigned short r) {
  v64u32 x_interleaved = Q6_W_vshuff_VVR(Q6_V_hi_W(x), Q6_V_lo_W(x), 4);
  v64u32 interleaved = Q6_Wuw_vmpyacc_WuwVuhRuh(x_interleaved, v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 4);
}

// out[i] = x[i] + (int32_t)((int16_t)v[i] * (int16_t)r)
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v64i32
ripple_ew_pure_hvx_i32_vmpyacc_i32i16i16_noshuff(v64i32 x, v64i16 v, short r) {
  v64i32 x_interleaved = Q6_W_vshuff_VVR(Q6_V_hi_W(x), Q6_V_lo_W(x), 4);
  v64i32 interleaved = Q6_Ww_vmpyacc_WwVhRh(x_interleaved, v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 4);
}

// out[i] = sat32(x[i] + (int32_t)((int16_t)v[i] * (int16_t)r))
// OUTPUT: sequential -- out[i] corresponds to input[i]
RIPPLE_INTRIN_INLINE v64i32
ripple_ew_pure_hvx_i32_vmpyacc_i32i16i16_sat_noshuff(v64i32 x, v64i16 v,
                                                     short r) {
  v64i32 x_interleaved = Q6_W_vshuff_VVR(Q6_V_hi_W(x), Q6_V_lo_W(x), 4);
  v64i32 interleaved = Q6_Ww_vmpyacc_WwVhRh_sat(x_interleaved, v, r);
  return Q6_W_vdeal_VVR(Q6_V_hi_W(interleaved), Q6_V_lo_W(interleaved), 4);
}

#ifdef __cplusplus
}
#endif
