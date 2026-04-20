//==============================================================================
//
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//

// HVX vector average (vavg) and negated average (vnavg) runtime library.
//
// Each function is a thin wrapper around a Hexagon HVX built-in intrinsic.
// RIPPLE_INTRIN_INLINE ensures the function is always inlined and emitted as
// a Ripple pure HVX intrinsic, allowing the Ripple compiler pass to recognize
// and replace calls with the corresponding HVX vector instruction.
//
// Naming convention:
//   ripple_ew_pure_hvx_<outType>_<op>_<inType0><inType1>
//   where <op> is one of:
//     vavg     - truncating average:        (a + b) >> 1
//     vavg_rnd - rounding average:          (a + b + 1) >> 1
//     vnavg    - negated average:           (a - b) >> 1

#include "hexagon_types.h"
#include "lib_func_attrib.h"
#ifndef RIPPLE_HVX_H
#define RIPPLE_HVX_H
#include <ripple_hvx.h>
#endif // RIPPLE_HVX_H
#ifdef __cplusplus
extern "C" {
#endif

//
// _____________________ v128i8 ____________________________
// 128 lanes of signed 8-bit integers (int8_t)
//

// Truncating average of two signed 8-bit vectors: out[i] = (a[i] + b[i]) >> 1
RIPPLE_INTRIN_INLINE v128i8 ripple_ew_pure_hvx_i8_vavg_i8i8(v128i8 a,
                                                            v128i8 b) {
  return Q6_Vb_vavg_VbVb(a, b);
}

// Rounding average of two signed 8-bit vectors: out[i] = (a[i] + b[i] + 1) >> 1
RIPPLE_INTRIN_INLINE v128i8 ripple_ew_pure_hvx_i8_vavg_i8i8_rnd(v128i8 a,
                                                                v128i8 b) {
  return Q6_Vb_vavg_VbVb_rnd(a, b);
}

// Negated average of two signed 8-bit vectors: out[i] = (a[i] - b[i]) >> 1
// See also: ripple_ew_pure_hvx_i8_vnavg_u8u8 for the unsigned input variant.
RIPPLE_INTRIN_INLINE v128i8 ripple_ew_pure_hvx_i8_vnavg_i8i8(v128i8 a,
                                                             v128i8 b) {
  return Q6_Vb_vnavg_VbVb(a, b);
}

//
// _____________________ v128u8 ____________________________
// 128 lanes of unsigned 8-bit integers (uint8_t)
//

// Truncating average of two unsigned 8-bit vectors: out[i] = (a[i] + b[i]) >> 1
RIPPLE_INTRIN_INLINE v128u8 ripple_ew_pure_hvx_u8_vavg_u8u8(v128u8 a,
                                                            v128u8 b) {
  return Q6_Vub_vavg_VubVub(a, b);
}

// Rounding average of two unsigned 8-bit vectors: out[i] = (a[i] + b[i] + 1) >>
// 1
RIPPLE_INTRIN_INLINE v128u8 ripple_ew_pure_hvx_u8_vavg_u8u8_rnd(v128u8 a,
                                                                v128u8 b) {
  return Q6_Vub_vavg_VubVub_rnd(a, b);
}

// Negated average of two unsigned 8-bit vectors, result is signed:
// out[i] = (a[i] - b[i]) >> 1
RIPPLE_INTRIN_INLINE v128i8 ripple_ew_pure_hvx_i8_vnavg_u8u8(v128u8 a,
                                                             v128u8 b) {
  return Q6_Vb_vnavg_VubVub(a, b);
}

//
// _____________________ v64u16 ____________________________
// 64 lanes of unsigned 16-bit integers (uint16_t)
//

// Truncating average of two unsigned 16-bit vectors: out[i] = (a[i] + b[i]) >>
// 1
RIPPLE_INTRIN_INLINE v64u16 ripple_ew_pure_hvx_u16_vavg_u16u16(v64u16 a,
                                                               v64u16 b) {
  return Q6_Vuh_vavg_VuhVuh(a, b);
}

// Rounding average of two unsigned 16-bit vectors: out[i] = (a[i] + b[i] + 1)
// >> 1
RIPPLE_INTRIN_INLINE v64u16 ripple_ew_pure_hvx_u16_vavg_u16u16_rnd(v64u16 a,
                                                                   v64u16 b) {
  return Q6_Vuh_vavg_VuhVuh_rnd(a, b);
}

//
// _____________________ v64i16 ____________________________
// 64 lanes of signed 16-bit integers (int16_t)
//

// Truncating average of two signed 16-bit vectors: out[i] = (a[i] + b[i]) >> 1
RIPPLE_INTRIN_INLINE v64i16 ripple_ew_pure_hvx_i16_vavg_i16i16(v64i16 a,
                                                               v64i16 b) {
  return Q6_Vh_vavg_VhVh(a, b);
}

// Rounding average of two signed 16-bit vectors: out[i] = (a[i] + b[i] + 1) >>
// 1
RIPPLE_INTRIN_INLINE v64i16 ripple_ew_pure_hvx_i16_vavg_i16i16_rnd(v64i16 a,
                                                                   v64i16 b) {
  return Q6_Vh_vavg_VhVh_rnd(a, b);
}

// Negated average of two signed 16-bit vectors: out[i] = (a[i] - b[i]) >> 1
RIPPLE_INTRIN_INLINE v64i16 ripple_ew_pure_hvx_i16_vnavg_i16i16(v64i16 a,
                                                                v64i16 b) {
  return Q6_Vh_vnavg_VhVh(a, b);
}

//
// _____________________ v32u32 ____________________________
// 32 lanes of unsigned 32-bit integers (uint32_t)
//

// Truncating average of two unsigned 32-bit vectors: out[i] = (a[i] + b[i]) >>
// 1
RIPPLE_INTRIN_INLINE v32u32 ripple_ew_pure_hvx_u32_vavg_u32u32(v32u32 a,
                                                               v32u32 b) {
  return Q6_Vuw_vavg_VuwVuw(a, b);
}

// Rounding average of two unsigned 32-bit vectors: out[i] = (a[i] + b[i] + 1)
// >> 1
RIPPLE_INTRIN_INLINE v32u32 ripple_ew_pure_hvx_u32_vavg_u32u32_rnd(v32u32 a,
                                                                   v32u32 b) {
  return Q6_Vuw_vavg_VuwVuw_rnd(a, b);
}

//
// _____________________ v32i32 ____________________________
// 32 lanes of signed 32-bit integers (int32_t)
//

// Truncating average of two signed 32-bit vectors: out[i] = (a[i] + b[i]) >> 1
RIPPLE_INTRIN_INLINE v32i32 ripple_ew_pure_hvx_i32_vavg_i32i32(v32i32 a,
                                                               v32i32 b) {
  return Q6_Vw_vavg_VwVw(a, b);
}

// Rounding average of two signed 32-bit vectors: out[i] = (a[i] + b[i] + 1) >>
// 1
RIPPLE_INTRIN_INLINE v32i32 ripple_ew_pure_hvx_i32_vavg_i32i32_rnd(v32i32 a,
                                                                   v32i32 b) {
  return Q6_Vw_vavg_VwVw_rnd(a, b);
}

// Negated average of two signed 32-bit vectors: out[i] = (a[i] - b[i]) >> 1
RIPPLE_INTRIN_INLINE v32i32 ripple_ew_pure_hvx_i32_vnavg_i32i32(v32i32 a,
                                                                v32i32 b) {
  return Q6_Vw_vnavg_VwVw(a, b);
}

#ifdef __cplusplus
}
#endif
