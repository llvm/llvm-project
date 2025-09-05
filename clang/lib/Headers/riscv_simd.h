/*===---- riscv_simd.h - RISC-V 'Packed SIMD' intrinsics --------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */


#ifndef __RISCV_PACKED_SIMD_H
#define __RISCV_PACKED_SIMD_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined (__riscv_p)

#if __riscv_xlen == 32
static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_b(uint32_t __x, int __y) {
  return __builtin_riscv_pslli_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_h(uint32_t __x, int __y) {
  return __builtin_riscv_pslli_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
  __riscv_psslai_h(uint32_t __x, int __y) {
  return __builtin_riscv_psslai_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_sslai(uint32_t __x, int __y) {
  return __builtin_riscv_sslai(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_bs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psll_bs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_hs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psll_hs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_bs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_padd_bs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_hs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_padd_hs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pusati_h(uint32_t __x, int __y) {
  return __builtin_riscv_pusati_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_usati(uint32_t __x, int __y) {
  return __builtin_riscv_usati_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrai_b(uint32_t __x, int __y) {
  return __builtin_riscv_psrai_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrai_h(uint32_t __x, int __y) {
  return __builtin_riscv_psrai_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrari_h(uint32_t __x, int __y) {
  return __builtin_riscv_psrari_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_srari(int32_t __x, int __y) {
  return __builtin_riscv_srari_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psati_h(uint32_t __x, int __y) {
  return __builtin_riscv_psati_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_sati(int32_t __x, int __y) {
  return __builtin_riscv_sati_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_bs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psrl_bs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_hs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psrl_hs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsum_bs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_predsum_bs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsum_hs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_predsum_hs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsumu_bs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_predsumu_bs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsumu_hs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_predsumu_hs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_bs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psra_bs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_hs(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psra_hs_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_padd_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_padd_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_sadd(int32_t __x, int32_t __y) {
  return __builtin_riscv_sadd(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psadd_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psadd_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psadd_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psadd_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_aadd(int32_t __x, int32_t __y) {
  return __builtin_riscv_aadd(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_paadd_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_paadd_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_paadd_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_paadd_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_saddu(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_saddu(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psaddu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psaddu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psaddu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psaddu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_aaddu(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_aaddu(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_paaddu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_paaddu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_paaddu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_paaddu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psub_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psub_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_ssub(int32_t __x, int32_t __y) {
  return __builtin_riscv_ssub(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssub_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pssub_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssub_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pssub_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_asub(int32_t __x, int32_t __y) {
  return __builtin_riscv_asub(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasub_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pasub_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasub_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pasub_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_ssubu(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_ssubu(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssubu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pssubu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssubu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pssubu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_asubu(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_asubu(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasubu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pasubu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasubu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pasubu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdif_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pdif_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdif_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pdif_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdifu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pdifu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdifu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pdifu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_h_b01(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmul_h_b01_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_h_b01(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulu_h_b01_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mul_h01(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mul_h01(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulu_h01(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulu_h01(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_slx(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_slx_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psh1add_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psh1add_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_ssh1sadd(int32_t __x, int32_t __y) {
  return __builtin_riscv_ssh1sadd(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssh1sadd_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pssh1sadd_h_32(__x, __y);
}
#endif


#if __riscv_xlen == 64
static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_b(uint64_t __x, int __y) {
  return __builtin_riscv_pslli_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_h(uint64_t __x, int __y) {
  return __builtin_riscv_pslli_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_w(uint64_t __x, int __y) {
  return __builtin_riscv_pslli_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psslai_h(uint64_t __x, int __y) {
  return __builtin_riscv_psslai_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psslai_w(uint64_t __x, int __y) {
  return __builtin_riscv_psslai_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_bs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psll_bs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_hs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psll_hs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_ws(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psll_ws(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_bs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_padd_bs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_hs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_padd_hs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_ws(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_padd_ws(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pusati_h(uint64_t __x, int __y) {
  return __builtin_riscv_pusati_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pusati_w(uint64_t __x, int __y) {
  return __builtin_riscv_pusati_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_usati(uint64_t __x, int __y) {
  return __builtin_riscv_usati_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrai_b(uint64_t __x, int __y) {
  return __builtin_riscv_psrai_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrai_h(uint64_t __x, int __y) {
  return __builtin_riscv_psrai_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrai_w(uint64_t __x, int __y) {
  return __builtin_riscv_psrai_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrari_h(uint64_t __x, int __y) {
  return __builtin_riscv_psrari_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrari_w(uint64_t __x, int __y) {
  return __builtin_riscv_psrari_w(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_srari(int64_t __x, int __y) {
  return __builtin_riscv_srari_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psati_h(uint64_t __x, int __y) {
  return __builtin_riscv_psati_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psati_w(uint64_t __x, int __y) {
  return __builtin_riscv_psati_w(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_sati(int64_t __x, int __y) {
  return __builtin_riscv_sati_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_bs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psrl_bs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_hs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psrl_hs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_ws(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psrl_ws(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsum_bs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_predsum_bs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsum_hs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_predsum_hs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsum_ws(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_predsum_ws(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsumu_bs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_predsumu_bs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsumu_hs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_predsumu_hs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_predsumu_ws(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_predsumu_ws(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_bs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psra_bs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_hs(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psra_hs_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_ws(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psra_ws(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_padd_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_padd_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_padd_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psadd_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psadd_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psadd_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psadd_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psadd_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psadd_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paadd_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paadd_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paadd_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paadd_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paadd_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paadd_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psaddu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psaddu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psaddu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psaddu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psaddu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psaddu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paaddu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paaddu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paaddu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paaddu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paaddu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paaddu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psub_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psub_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psub_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssub_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssub_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssub_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssub_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssub_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssub_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasub_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasub_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasub_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasub_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasub_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasub_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssubu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssubu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssubu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssubu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssubu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssubu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasubu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasubu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasubu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasubu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasubu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasubu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdif_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pdif_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdif_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pdif_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdifu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pdifu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pdifu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pdifu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_h_b01(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmul_h_b01_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_w_h01(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmul_w_h01(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_h_b01(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulu_h_b01_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_w_h01(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulu_w_h01(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mul_w01(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mul_w01(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulu_w01(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mulu_w01(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_slx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_slx_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psh1add_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psh1add_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psh1add_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psh1add_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssh1sadd_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssh1sadd_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssh1sadd_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssh1sadd_w(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_unzip8p(int64_t __x, int64_t __y) {
  return __builtin_riscv_unzip8p(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_unzip16p(int64_t __x, int64_t __y) {
  return __builtin_riscv_unzip16p(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_unzip8hp(int64_t __x, int64_t __y) {
  return __builtin_riscv_unzip8hp(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_unzip16hp(int64_t __x, int64_t __y) {
  return __builtin_riscv_unzip16hp(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_zip8p(int64_t __x, int64_t __y) {
  return __builtin_riscv_zip8p(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_zip16p(int64_t __x, int64_t __y) {
  return __builtin_riscv_zip16p(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_zip8hp(int64_t __x, int64_t __y) {
  return __builtin_riscv_zip8hp(__x, __y);
}

static __inline__ int64_t __attribute__((__always_inline__, __nodebug__))
__riscv_zip16hp(int64_t __x, int64_t __y) {
  return __builtin_riscv_zip16hp(__x, __y);
}
#endif

#endif // defined(__riscv_p)


#if defined(__cplusplus)
}
#endif

#endif
