//===------------------------- HVX_Narrow_Shift.cc-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===
//                        Ripple Narrow Shift Library
//===------------------------------------------------------------------------===

#include "lib_func_attrib.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <ripple_hvx.h>

extern "C" {

// i32 -> i16
RIPPLE_INTRIN_INLINE HVX_Vector ripple_hvx_narsh_i32toi16(HVX_Vector Vec_Odd,
                                                          HVX_Vector Vec_Even,
                                                          uint32_t Shift) {
  return Q6_Vh_vasr_VwVwR(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector ripple_hvx_narsh_sat_i32toi16(
    HVX_Vector Vec_Odd, HVX_Vector Vec_Even, uint32_t Shift) {
  return Q6_Vh_vasr_VwVwR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector ripple_hvx_narsh_rnd_sat_i32toi16(
    HVX_Vector Vec_Odd, HVX_Vector Vec_Even, uint32_t Shift) {
  return Q6_Vh_vasr_VwVwR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector
ripple_mask_hvx_narsh_i32toi16(HVX_Vector Vec_Odd, HVX_Vector Vec_Even,
                               uint32_t Shift, HVX_VectorPred mask) {
  (void)mask;
  return Q6_Vh_vasr_VwVwR(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector
ripple_mask_hvx_narsh_sat_i32toi16(HVX_Vector Vec_Odd, HVX_Vector Vec_Even,
                                   uint32_t Shift, HVX_VectorPred mask) {
  (void)mask;
  return Q6_Vh_vasr_VwVwR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector
ripple_mask_hvx_narsh_rnd_sat_i32toi16(HVX_Vector Vec_Odd, HVX_Vector Vec_Even,
                                       uint32_t Shift, HVX_VectorPred mask) {
  (void)mask;
  return Q6_Vh_vasr_VwVwR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

// i32 -> u16
RIPPLE_INTRIN_INLINE HVX_Vector ripple_hvx_narsh_sat_i32tou16(
    HVX_Vector Vec_Odd, HVX_Vector Vec_Even, uint32_t Shift) {
  return Q6_Vuh_vasr_VwVwR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector ripple_hvx_narsh_rnd_sat_i32tou16(
    HVX_Vector Vec_Odd, HVX_Vector Vec_Even, uint32_t Shift) {
  return Q6_Vuh_vasr_VwVwR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

RIPPLE_INTRIN_INLINE HVX_Vector
ripple_mask_hvx_narsh_sat_i32tou16(HVX_Vector Vec_Odd, HVX_Vector Vec_Even,
                                   uint32_t Shift, HVX_VectorPred mask) {
  (void)mask;
  return Q6_Vuh_vasr_VwVwR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector
ripple_mask_hvx_narsh_rnd_sat_i32tou16(HVX_Vector Vec_Odd, HVX_Vector Vec_Even,
                                       uint32_t Shift, HVX_VectorPred mask) {
  (void)mask;
  return Q6_Vuh_vasr_VwVwR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

// u32 -> u16
RIPPLE_INTRIN_INLINE HVX_Vector ripple_hvx_narsh_sat_u32tou16(
    HVX_Vector Vec_Odd, HVX_Vector Vec_Even, uint32_t Shift) {
  return Q6_Vuh_vasr_VuwVuwR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector ripple_hvx_narsh_rnd_sat_u32tou16(
    HVX_Vector Vec_Odd, HVX_Vector Vec_Even, uint32_t Shift) {
  return Q6_Vuh_vasr_VuwVuwR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

RIPPLE_INTRIN_INLINE HVX_Vector
ripple_mask_hvx_narsh_sat_u32tou16(HVX_Vector Vec_Odd, HVX_Vector Vec_Even,
                                   uint32_t Shift, HVX_VectorPred mask) {
  (void)mask;
  return Q6_Vuh_vasr_VuwVuwR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE HVX_Vector
ripple_mask_hvx_narsh_rnd_sat_u32tou16(HVX_Vector Vec_Odd, HVX_Vector Vec_Even,
                                       uint32_t Shift, HVX_VectorPred mask) {
  (void)mask;
  return Q6_Vuh_vasr_VuwVuwR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

// i16 -> i8
RIPPLE_INTRIN_INLINE v64i16 ripple_hvx_narsh_sat_i16toi8(v64i16 Vec_Odd,
                                                         v64i16 Vec_Even,
                                                         uint32_t Shift) {
  return Q6_Vb_vasr_VhVhR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE v64i16 ripple_hvx_narsh_rnd_sat_i16toi8(v64i16 Vec_Odd,
                                                             v64i16 Vec_Even,
                                                             uint32_t Shift) {
  return Q6_Vb_vasr_VhVhR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

RIPPLE_INTRIN_INLINE v64i16 ripple_mask_hvx_narsh_sat_i16toi8(v64i16 Vec_Odd,
                                                              v64i16 Vec_Even,
                                                              uint32_t Shift,
                                                              v64i16 mask) {
  (void)mask;
  return Q6_Vb_vasr_VhVhR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE v64i16 ripple_mask_hvx_narsh_rnd_sat_i16toi8(
    v64i16 Vec_Odd, v64i16 Vec_Even, uint32_t Shift, v64i16 mask) {
  (void)mask;
  return Q6_Vb_vasr_VhVhR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

// i16 -> u8
RIPPLE_INTRIN_INLINE v64i16 ripple_hvx_narsh_sat_i16tou8(v64i16 Vec_Odd,
                                                         v64i16 Vec_Even,
                                                         uint32_t Shift) {
  return Q6_Vub_vasr_VhVhR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE v64i16 ripple_hvx_narsh_rnd_sat_i16tou8(v64i16 Vec_Odd,
                                                             v64i16 Vec_Even,
                                                             uint32_t Shift) {
  return Q6_Vub_vasr_VhVhR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

RIPPLE_INTRIN_INLINE v64i16 ripple_mask_hvx_narsh_sat_i16tou8(v64i16 Vec_Odd,
                                                              v64i16 Vec_Even,
                                                              uint32_t Shift,
                                                              v64i16 mask) {
  (void)mask;
  return Q6_Vub_vasr_VhVhR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE v64i16 ripple_mask_hvx_narsh_rnd_sat_i16tou8(
    v64i16 Vec_Odd, v64i16 Vec_Even, uint32_t Shift, v64i16 mask) {
  (void)mask;
  return Q6_Vub_vasr_VhVhR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

// u16 -> u8
RIPPLE_INTRIN_INLINE v64i16 ripple_hvx_narsh_sat_u16tou8(v64i16 Vec_Odd,
                                                         v64i16 Vec_Even,
                                                         uint32_t Shift) {
  return Q6_Vub_vasr_VuhVuhR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE v64i16 ripple_hvx_narsh_rnd_sat_u16tou8(v64i16 Vec_Odd,
                                                             v64i16 Vec_Even,
                                                             uint32_t Shift) {
  return Q6_Vub_vasr_VuhVuhR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE v64i16 ripple_mask_hvx_narsh_sat_u16tou8(v64i16 Vec_Odd,
                                                              v64i16 Vec_Even,
                                                              uint32_t Shift,
                                                              v64i16 mask) {
  (void)mask;
  return Q6_Vub_vasr_VuhVuhR_sat(Vec_Odd, Vec_Even, Shift);
}
RIPPLE_INTRIN_INLINE v64i16 ripple_mask_hvx_narsh_rnd_sat_u16tou8(
    v64i16 Vec_Odd, v64i16 Vec_Even, uint32_t Shift, v64i16 mask) {
  (void)mask;
  return Q6_Vub_vasr_VuhVuhR_rnd_sat(Vec_Odd, Vec_Even, Shift);
}

} // extern "C"
