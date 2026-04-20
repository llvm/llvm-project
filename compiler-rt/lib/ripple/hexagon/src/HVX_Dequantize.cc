//==============================================================================
//
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// Part of the Ripple vector library to support the HVX Dequantize
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

RIPPLE_INTRIN_INLINE v128u32 ripple_pure_ew_hvx_dequantize_u8_to_f32(
    v128u8 vin, const int16_t zero_offset, const uint8_t from_qint8,
    const int32_t iscale, const int16_t exp_base) {
  auto BS = ripple_set_block_shape(0, 128);
  uint8_t in = vec_to_ripple<128, uint8_t>(BS, vin);

  uint8_t in_ub = in ^ from_qint8;

  // (1)  subtract the quantized zero; result is -255,255 in 16 bits.
  int16_t vin_h = static_cast<int16_t>(in_ub) - zero_offset;

  // (2)  take abs. value, multiply by a normalized 24 bit scale. This can be
  // done
  //      using vmpyie, vmpyio, and the results barely fit in 32 bits unsigned.
  int16_t vin_h_neg = -vin_h;
  int16_t inabs_h = vin_h > 0 ? vin_h : vin_h_neg;
  int32_t prod_w = iscale * static_cast<int32_t>(inabs_h);

  // (3)  find the 'cl0' which is in 0..8 except when result from (1) is zero.
  // product is zero iff vin_h =0;
  // since scale is 0x800000 ... 0xFFFFFF,
  // for vin_h in 1...255, prod is 0x0080:0000 ... 0xFEFF:FF01
  // so clz is 0..8
  uint32_t clz = __builtin_clz(prod_w);
  uint16_t clz_h = static_cast<uint16_t>(clz & 0xFFFF);

  // (4)  << by cl0, so that MSB is 1; then add 0x80, and >> by 8 to get 24-bit
  // rounded
  //      mantissa. Overflow is possible in the +0x80: e.g. 0x81*0xfe03f8 ->
  //      0x7ffffff8 which is 0xfffffff0 after normalize, and 0x00000070 after
  //      +0x80. This is detected by the msb being zero after +0x80; we add 1 to
  //      exponent to fix (this is only possible when cl0 is >=1, <=7, so the
  //      final exponent is never more than exp_base).
  prod_w = prod_w << clz_h;

  // for rounding: add 0x80 then >>8.
  // if result after add has 0 in msb, then overflow has occurred; for that we
  // need to bump the exponent. We can leave upper 24 bits as zero -  it should
  // change to 0x800000xx, but it doesn't matter since bit 31 will be lost
  // anyway.

  int32_t mant = prod_w + 0x80;
  int16_t manth = static_cast<int16_t>(mant >> 16); // collect upper to h
  mant = mant >> 8;

  // (5)  pack the shift amounts back down to 16-bits, so they can be used to
  // form the exponent and
  //      combined with signs from (1). This step also detects zeros from (1).

  // generate [sign:exp:0000000]
  // and transfer the sign from vin_h
  uint16_t sign = vin_h & 0x8000;
  uint16_t base_with_sign = (exp_base * 128) | (sign ? 0x8000 : 0);
  // move clz counts up 7 bits, subtract from basexp*128
  uint16_t exp = base_with_sign - clz_h * 0x0080;

  // overflow if manth are >= 0( bit 7 clear); add 1(<<7) to exp
  bool noflo = (0 > manth) ? true : false;
  bool iszero = (vin_h == 0) ? true : false;

  uint16_t oflo_add = (!noflo) ? 0x0080 : 0;
  exp += oflo_add;
  // that's the exponent almost done. clear if value is 0.
  exp = (!iszero) ? exp : 0;

  // (6)  combine the exponents into results from 4.
  // now, exp_02 contains  s:eeeeeeee:0000000
  // and we need to merge that to 'mant' values
  uint32_t exp_w = static_cast<uint32_t>(exp << 16);
  uint32_t res = (mant & 0x007FFFFF) | exp_w;

  return ripple_to_vec<128, uint32_t>(BS, res);
}

RIPPLE_INTRIN_INLINE v64f32 ripple_pure_hvx_dequantize_u16_to_f32_flat(
    const v64u16 vin, uint32_t offset, float scale) {
  // cast input up to uw32
  auto BS = ripple_set_block_shape(0, 64);
  uint16_t in16 = vec_to_ripple<64, uint16_t>(BS, vin);
  int32_t in32 = static_cast<int32_t>(in16);
  int32_t sub32 = in32 - offset;
  float sub_f = static_cast<float>(sub32);
  return ripple_to_vec<64, float>(BS, sub_f * scale);
}

RIPPLE_INTRIN_INLINE v64u32 ripple_pure_hvx_dequantize_u16_to_f32(
    const v64u16 vin, const uint16_t zero_offset, const uint16_t from_qint16,
    const int32_t iscale, const int16_t exp_base) {

  auto BS = ripple_set_block_shape(0, 64);
  uint16_t in = vec_to_ripple<64, uint16_t>(BS, vin);

  // (1) make u16 range, if not.
  uint16_t in_uh = in ^ from_qint16;
  // (2) abs diff.
  uint16_t rzero_offset = ripple_broadcast(BS, 1, zero_offset);
  v64u16 vin_uh = ripple_to_vec<64, uint16_t>(BS, in_uh);
  v64u16 vzero_offset = ripple_to_vec<64, uint16_t>(BS, rzero_offset);
  v64u16 vmag_uh = Q6_Vuh_vabsdiff_VuhVuh(vin_uh, vzero_offset);
  uint16_t mag_uh = vec_to_ripple<64, uint16_t>(BS, vmag_uh);
  // == 0?
  uint16_t q_is_zero_h = (zero_offset == in_uh) ? 1 : 0;
  // > 0?
  // pack the signs for later, into bit 15 of each h lane
  // signs in bit 15
  uint16_t sign_uh = (zero_offset > in_uh) ? 0x8000 : 0x0000;

  // (3) normalize the uh mag.
  // v64u16 vmag_uh = ripple_to_vec<64, uint16_t>(BS, mag_uh);
  // v64u16 vnormshift_uh = Q6_Vuh_vcl0_Vuh(vmag_uh);
  // uint16_t normshift_uh = vec_to_ripple<64, uint16_t>(BS, vnormshift_uh);
  uint16_t normshift_uh = __builtin_clz(mag_uh << 16);
  // now 0x8000 .. 0xFFFF (assuming nonzero)
  mag_uh = mag_uh << normshift_uh;
  // account the exponents.
  int16_t exp_h = exp_base - normshift_uh;

  // (4) now do the vmpye. Only the even uh lanes of the second input to this op
  // are used.
  int32_t mag_w = static_cast<int32_t>(
      (static_cast<int64_t>(iscale) * static_cast<int64_t>(mag_uh)) >> 16);

  // (5) determine whether to vmpyo by 256 or 512 (small = 0 or 1)
  int32_t small_w = (0x3fffffe0 > mag_w) ? 1 : 0;
  int32_t scale_w = 0x01000000 << small_w;

  v64i32 vscale_w = ripple_to_vec<64, int32_t>(BS, scale_w);
  v64i32 vmag_w = ripple_to_vec<64, int32_t>(BS, mag_w);
  v32i32 vmant0_w =
      Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(vmag_w), Q6_V_lo_W(vscale_w));
  v32i32 vmant1_w =
      Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(vmag_w), Q6_V_hi_W(vscale_w));
  v64i32 vmant_w = Q6_W_vcombine_VV(vmant1_w, vmant0_w);
  int32_t mant_w = vec_to_ripple<64, int32_t>(BS, vmant_w);
  int32_t k_007F_FFFF = 0x007FFFFF;
  mant_w = mant_w & ripple_broadcast(BS, 1, k_007F_FFFF);

  // make a zero mask in h : 0 when input is zero, 0xFFFF when not.
  int16_t nzmask_h = q_is_zero_h == 1 ? 1 : -1;

  // (6) pack small0_w, small1_w together and subtract from the exponent; this
  // gives the final exponent field.
  exp_h -= small_w;
  // don't allow it > 0xFF. If it's >= 0xFF that's overflow and we want to force
  // mantissa bits to zero, so we get an infinity.
  exp_h = (exp_h < 0x00FF) ? exp_h : 0x00FF;
  // drive exp to zero if the value is zero
  exp_h = exp_h & nzmask_h;
  // now, clear the zero mask if exponent overflow occurred (to get inf)
  int32_t nzmask_w = (0x00FF > exp_h) ? nzmask_h : 0;

  // (7) << by 7 to place exponent, then add in the sign. Result: [ sign : exp :
  // 0000000] split the exponent out... upper part of each w wlane, 0 in the
  // rest
  uint32_t exp_w = ((exp_h << 7) + sign_uh) << 16;

  uint32_t result = (nzmask_w & mant_w) | exp_w;

  return ripple_to_vec<64, uint32_t>(BS, result);
}

#ifdef __cplusplus
}
#endif
