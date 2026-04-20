//==============================================================================
//
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// Part of the Ripple vector library to support the HVX Quantize
// instructions.
//
//==============================================================================
//

#include "lib_func_attrib.h"
#include <cstdint>
#include <cstring>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <ripple/zip.h>
#ifndef RIPPLE_HVX_H
#define RIPPLE_HVX_H
#include <ripple_hvx.h>
#endif // RIPPLE_HVX_H
#ifdef __cplusplus
extern "C" {
#endif

RIPPLE_INTRIN_INLINE v128u8 ripple_pure_ew_hvx_quantize_f32_to_u8(
    float scale_f, int16_t out_offset, const v128f32 in) {
  // return quantize_sf_to_u8(in, scale_f, out_offset);
  auto BS = ripple_set_block_shape(0, 128);
  float xf = vec_to_ripple<128, float>(BS, in);

  xf = xf * scale_f;

  _Float16 xhf = static_cast<_Float16>(xf);

  int16_t xh = static_cast<int16_t>(xhf);

  xh = __builtin_ripple_add_sat_i16(xh, out_offset);

  v128u16 vh = ripple_to_vec<128, uint16_t>(BS, xh);
  // return ripple_ew_pure_u8_vpack_u16_sat(vh);
  return Q6_Vub_vpack_VhVh_sat(Q6_V_hi_W(vh), Q6_V_lo_W(vh));
}

RIPPLE_INTRIN_INLINE v64u16 ripple_pure_ew_hvx_quantize_f32_to_u16(
    float scale_f, int32_t out_offset, const v64f32 in) {
  auto BS = ripple_set_block_shape(0, 64);
  float xf = vec_to_ripple<64, float>(BS, in);

  xf = xf * scale_f;

  uint32_t xf_bits;
  std::memcpy(&xf_bits, &xf, sizeof(xf));
  int32_t sign_bits = xf_bits & 0x80000000;
  sign_bits = sign_bits | 0x3F000000;
  float sign_f;
  std::memcpy(&sign_f, &sign_bits, sizeof(sign_bits));
  xf = xf + sign_f;

  int32_t xw = static_cast<int32_t>(xf);

  xw = __builtin_ripple_add_sat_i32(xw, out_offset);

  v64u32 vw = ripple_to_vec<64, uint32_t>(BS, xw);
  // return ripple_ew_pure_u16_vpack_i32_sat(vw);
  return Q6_Vuh_vpack_VwVw_sat(Q6_V_hi_W(vw), Q6_V_lo_W(vw));
}
#ifdef __cplusplus
}
#endif
