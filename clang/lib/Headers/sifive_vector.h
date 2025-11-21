//===----- sifive_vector.h - SiFive Vector definitions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _SIFIVE_VECTOR_H_
#define _SIFIVE_VECTOR_H_

#include "riscv_vector.h"

#pragma clang riscv intrinsic sifive_vector

#define __riscv_sf_vc_x_se_u8mf4(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint8_t)rs1, 8, 6, vl)
#define __riscv_sf_vc_x_se_u8mf2(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint8_t)rs1, 8, 7, vl)
#define __riscv_sf_vc_x_se_u8m1(p27_26, p24_20, p11_7, rs1, vl)                \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint8_t)rs1, 8, 0, vl)
#define __riscv_sf_vc_x_se_u8m2(p27_26, p24_20, p11_7, rs1, vl)                \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint8_t)rs1, 8, 1, vl)
#define __riscv_sf_vc_x_se_u8m4(p27_26, p24_20, p11_7, rs1, vl)                \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint8_t)rs1, 8, 2, vl)
#define __riscv_sf_vc_x_se_u8m8(p27_26, p24_20, p11_7, rs1, vl)                \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint8_t)rs1, 8, 3, vl)

#define __riscv_sf_vc_x_se_u16mf2(p27_26, p24_20, p11_7, rs1, vl)              \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint16_t)rs1, 16, 7, vl)
#define __riscv_sf_vc_x_se_u16m1(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint16_t)rs1, 16, 0, vl)
#define __riscv_sf_vc_x_se_u16m2(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint16_t)rs1, 16, 1, vl)
#define __riscv_sf_vc_x_se_u16m4(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint16_t)rs1, 16, 2, vl)
#define __riscv_sf_vc_x_se_u16m8(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint16_t)rs1, 16, 3, vl)

#define __riscv_sf_vc_x_se_u32m1(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint32_t)rs1, 32, 0, vl)
#define __riscv_sf_vc_x_se_u32m2(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint32_t)rs1, 32, 1, vl)
#define __riscv_sf_vc_x_se_u32m4(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint32_t)rs1, 32, 2, vl)
#define __riscv_sf_vc_x_se_u32m8(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint32_t)rs1, 32, 3, vl)

#define __riscv_sf_vc_i_se_u8mf4(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 8, 7, vl)
#define __riscv_sf_vc_i_se_u8mf2(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 8, 6, vl)
#define __riscv_sf_vc_i_se_u8m1(p27_26, p24_20, p11_7, simm5, vl)              \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 8, 0, vl)
#define __riscv_sf_vc_i_se_u8m2(p27_26, p24_20, p11_7, simm5, vl)              \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 8, 1, vl)
#define __riscv_sf_vc_i_se_u8m4(p27_26, p24_20, p11_7, simm5, vl)              \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 8, 2, vl)
#define __riscv_sf_vc_i_se_u8m8(p27_26, p24_20, p11_7, simm5, vl)              \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 8, 3, vl)

#define __riscv_sf_vc_i_se_u16mf2(p27_26, p24_20, p11_7, simm5, vl)            \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 16, 7, vl)
#define __riscv_sf_vc_i_se_u16m1(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 16, 0, vl)
#define __riscv_sf_vc_i_se_u16m2(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 16, 1, vl)
#define __riscv_sf_vc_i_se_u16m4(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 16, 2, vl)
#define __riscv_sf_vc_i_se_u16m8(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 16, 3, vl)

#define __riscv_sf_vc_i_se_u32m1(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 32, 0, vl)
#define __riscv_sf_vc_i_se_u32m2(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 32, 1, vl)
#define __riscv_sf_vc_i_se_u32m4(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 32, 2, vl)
#define __riscv_sf_vc_i_se_u32m8(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 32, 3, vl)

#if __riscv_v_elen >= 64
#define __riscv_sf_vc_x_se_u8mf8(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint8_t)rs1, 8, 5, vl)
#define __riscv_sf_vc_x_se_u16mf4(p27_26, p24_20, p11_7, rs1, vl)              \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint16_t)rs1, 16, 6, vl)
#define __riscv_sf_vc_x_se_u32mf2(p27_26, p24_20, p11_7, rs1, vl)              \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint32_t)rs1, 32, 7, vl)

#define __riscv_sf_vc_i_se_u8mf8(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 8, 5, vl)
#define __riscv_sf_vc_i_se_u16mf4(p27_26, p24_20, p11_7, simm5, vl)            \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 16, 6, vl)
#define __riscv_sf_vc_i_se_u32mf2(p27_26, p24_20, p11_7, simm5, vl)            \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 32, 7, vl)

#define __riscv_sf_vc_i_se_u64m1(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 64, 0, vl)
#define __riscv_sf_vc_i_se_u64m2(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 64, 1, vl)
#define __riscv_sf_vc_i_se_u64m4(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 64, 2, vl)
#define __riscv_sf_vc_i_se_u64m8(p27_26, p24_20, p11_7, simm5, vl)             \
  __riscv_sf_vc_i_se(p27_26, p24_20, p11_7, simm5, 64, 3, vl)

#if __riscv_xlen >= 64
#define __riscv_sf_vc_x_se_u64m1(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint64_t)rs1, 64, 0, vl)
#define __riscv_sf_vc_x_se_u64m2(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint64_t)rs1, 64, 1, vl)
#define __riscv_sf_vc_x_se_u64m4(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint64_t)rs1, 64, 2, vl)
#define __riscv_sf_vc_x_se_u64m8(p27_26, p24_20, p11_7, rs1, vl)               \
  __riscv_sf_vc_x_se(p27_26, p24_20, p11_7, (uint64_t)rs1, 64, 3, vl)
#endif
#endif

#define __riscv_sf_vsettnt_e8w1(atn) __riscv_sf_vsettnt(atn, 0, 1);
#define __riscv_sf_vsettnt_e8w2(atn) __riscv_sf_vsettnt(atn, 0, 2);
#define __riscv_sf_vsettnt_e8w4(atn) __riscv_sf_vsettnt(atn, 0, 3);
#define __riscv_sf_vsettnt_e16w1(atn) __riscv_sf_vsettnt(atn, 1, 1);
#define __riscv_sf_vsettnt_e16w2(atn) __riscv_sf_vsettnt(atn, 1, 2);
#define __riscv_sf_vsettnt_e16w4(atn) __riscv_sf_vsettnt(atn, 1, 3);
#define __riscv_sf_vsettnt_e32w1(atn) __riscv_sf_vsettnt(atn, 2, 1);
#define __riscv_sf_vsettnt_e32w2(atn) __riscv_sf_vsettnt(atn, 2, 2);
#define __riscv_sf_vsettm_e8w1(atm) __riscv_sf_vsettm(atm, 0, 1);
#define __riscv_sf_vsettm_e8w2(atm) __riscv_sf_vsettm(atm, 0, 2);
#define __riscv_sf_vsettm_e8w4(atm) __riscv_sf_vsettm(atm, 0, 3);
#define __riscv_sf_vsettm_e16w1(atm) __riscv_sf_vsettm(atm, 1, 1);
#define __riscv_sf_vsettm_e16w2(atm) __riscv_sf_vsettm(atm, 1, 2);
#define __riscv_sf_vsettm_e16w4(atm) __riscv_sf_vsettm(atm, 1, 3);
#define __riscv_sf_vsettm_e32w1(atm) __riscv_sf_vsettm(atm, 2, 1);
#define __riscv_sf_vsettm_e32w2(atm) __riscv_sf_vsettm(atm, 2, 2);
#define __riscv_sf_vsettn_e8w1(atn) __riscv_sf_vsettn(atn, 0, 1);
#define __riscv_sf_vsettn_e8w2(atn) __riscv_sf_vsettn(atn, 0, 2);
#define __riscv_sf_vsettn_e8w4(atn) __riscv_sf_vsettn(atn, 0, 3);
#define __riscv_sf_vsettn_e16w1(atn) __riscv_sf_vsettn(atn, 1, 1);
#define __riscv_sf_vsettn_e16w2(atn) __riscv_sf_vsettn(atn, 1, 2);
#define __riscv_sf_vsettn_e16w4(atn) __riscv_sf_vsettn(atn, 1, 3);
#define __riscv_sf_vsettn_e32w1(atn) __riscv_sf_vsettn(atn, 2, 1);
#define __riscv_sf_vsettn_e32w2(atn) __riscv_sf_vsettn(atn, 2, 2);
#define __riscv_sf_vsettk_e8w1(atk) __riscv_sf_vsettk(atk, 0, 1);
#define __riscv_sf_vsettk_e8w2(atk) __riscv_sf_vsettk(atk, 0, 2);
#define __riscv_sf_vsettk_e8w4(atk) __riscv_sf_vsettk(atk, 0, 3);
#define __riscv_sf_vsettk_e16w1(atk) __riscv_sf_vsettk(atk, 1, 1);
#define __riscv_sf_vsettk_e16w2(atk) __riscv_sf_vsettk(atk, 1, 2);
#define __riscv_sf_vsettk_e16w4(atk) __riscv_sf_vsettk(atk, 1, 3);
#define __riscv_sf_vsettk_e32w1(atk) __riscv_sf_vsettk(atk, 2, 1);
#define __riscv_sf_vsettk_e32w2(atk) __riscv_sf_vsettk(atk, 2, 2);
#define __riscv_sf_vtzero_t_e8w1(tile, atm, atn)                               \
  __riscv_sf_vtzero_t(tile, atm, atn, 3, 1);
#define __riscv_sf_vtzero_t_e8w2(tile, atm, atn)                               \
  __riscv_sf_vtzero_t(tile, atm, atn, 3, 2);
#define __riscv_sf_vtzero_t_e8w4(tile, atm, atn)                               \
  __riscv_sf_vtzero_t(tile, atm, atn, 3, 4);
#define __riscv_sf_vtzero_t_e16w1(tile, atm, atn)                              \
  __riscv_sf_vtzero_t(tile, atm, atn, 4, 1);
#define __riscv_sf_vtzero_t_e16w2(tile, atm, atn)                              \
  __riscv_sf_vtzero_t(tile, atm, atn, 4, 2);
#define __riscv_sf_vtzero_t_e16w4(tile, atm, atn)                              \
  __riscv_sf_vtzero_t(tile, atm, atn, 4, 4);
#define __riscv_sf_vtzero_t_e32w1(tile, atm, atn)                              \
  __riscv_sf_vtzero_t(tile, atm, atn, 5, 1);
#define __riscv_sf_vtzero_t_e32w2(tile, atm, atn)                              \
  __riscv_sf_vtzero_t(tile, atm, atn, 5, 2);
#if __riscv_v_elen >= 64
#define __riscv_sf_vsettnt_e64w1(atn) __riscv_sf_vsettnt(atn, 3, 1);
#define __riscv_sf_vsettm_e64w1(atm) __riscv_sf_vsettm(atm, 3, 1);
#define __riscv_sf_vsettn_e64w1(atn) __riscv_sf_vsettn(atn, 3, 1);
#define __riscv_sf_vsettk_e64w1(atk) __riscv_sf_vsettk(atk, 3, 1);
#define __riscv_sf_vtzero_t_e64w1(tile, atm, atn)                              \
  __riscv_sf_vtzero_t(tile, atm, atn, 6, 1);
#endif
#endif //_SIFIVE_VECTOR_H_
