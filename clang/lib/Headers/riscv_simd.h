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

#if defined(__riscv_p)

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

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_h_b00(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmul_h_b00_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_h_b11(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmul_h_b11_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_h_b00(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulu_h_b00_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_h_b11(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulu_h_b11_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulsu_h_b00(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulsu_h_b00_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulsu_h_b11(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulsu_h_b11_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mul_h00(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mul_h00(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mul_h11(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mul_h11(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulu_h00(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulu_h00(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulu_h11(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulu_h11(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulsu_h00(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulsu_h00(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulsu_h11(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulsu_h11(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppack_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_ppack_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppackbt_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_ppackbt_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_packbt(int32_t __x, int32_t __y) {
  return __builtin_riscv_packbt_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppacktb_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_ppacktb_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_packtb(int32_t __x, int32_t __y) {
  return __builtin_riscv_packtb_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppackt_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_ppackt_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_packt(int32_t __x, int32_t __y) {
  return __builtin_riscv_packt_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pas_hx(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pas_hx_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psa_hx(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psa_hx_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_psas_hx(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_psas_hx_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssa_hx(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pssa_hx_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_paas_hx(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_paas_hx_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasa_hx(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pasa_hx_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mseq(int32_t __x, int32_t __y) {
  return __builtin_riscv_mseq(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmseq_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmseq_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmseq_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmseq_h_32(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mslt(int32_t __x, int32_t __y) {
  return __builtin_riscv_mslt(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmslt_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmslt_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmslt_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmslt_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_msltu(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_msltu(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmsltu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmsltu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmsltu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmsltu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmin_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmin_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmin_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmin_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pminu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pminu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pminu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pminu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmax_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmax_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmax_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmax_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmaxu_b(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmaxu_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmaxu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmaxu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulh_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_h_b0(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulh_h_b0_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_h_b1(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulh_h_b1_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulhu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhr_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulhr_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhru_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulhru_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulhsu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_h_b0(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulhsu_h_b0_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_h_b1(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulhsu_h_b1_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhrsu_h(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_pmulhrsu_h_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulh_h1(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulh_h1(__x, __y);
}

static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulhr(int32_t __x, int32_t __y) {
  return __builtin_riscv_mulhr(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulhru(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulhru(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulh_h0(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulh_h0(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulhsu_h0(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulhsu_h0(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulhsu_h1(uint32_t __x, uint32_t __y) {
  return __builtin_riscv_mulhsu_h1(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulhrsu(uint32_t __x, int32_t __y) {
  return __builtin_riscv_mulhrsu(__x, __y);
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

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_h_b00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmul_h_b00_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_w_h00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmul_w_h00(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_h_b11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmul_h_b11_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmul_w_h11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmul_w_h11(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_h_b00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulu_h_b00_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_w_h00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulu_w_h00(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_h_b11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulu_h_b11_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulu_w_h11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulu_w_h11(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulsu_h_b00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulsu_h_b00_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulsu_w_h00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulsu_w_h00(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulsu_h_b11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulsu_h_b11_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulsu_w_h11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulsu_w_h11(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mul_w00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mul_w00(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mul_w11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mul_w11(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulu_w00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mulu_w00(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulu_w11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mulu_w11(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulsu_w00(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mulsu_w00(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_mulsu_w11(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_mulsu_w11(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppack_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppack_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppack_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppack_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppackbt_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppackbt_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppackbt_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppackbt_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_packbt(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_packbt_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppacktb_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppacktb_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppacktb_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppacktb_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_packtb(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_packtb_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppackt_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppackt_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_ppackt_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_ppackt_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_packt(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_packt_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pas_hx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pas_hx_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pas_wx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pas_wx(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psa_hx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psa_hx_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psa_wx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psa_wx(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psas_hx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psas_hx_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_psas_wx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_psas_wx(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssa_hx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssa_hx_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pssa_wx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pssa_wx(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paas_hx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paas_hx_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_paas_wx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_paas_wx(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasa_hx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasa_hx_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pasa_wx(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pasa_wx(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmseq_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmseq_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmseq_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmseq_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmseq_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmseq_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmslt_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmslt_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmslt_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmslt_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmslt_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmslt_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmsltu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmsltu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmsltu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmsltu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmsltu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmsltu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmin_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmin_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmin_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmin_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmin_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmin_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pminu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pminu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pminu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pminu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pminu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pminu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmax_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmax_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmax_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmax_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmax_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmax_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmaxu_b(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmaxu_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmaxu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmaxu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmaxu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmaxu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulh_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulh_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_h_b0(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulh_h_b0_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_w_h0(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulh_w_h0(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_h_b1(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulh_h_b1_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulh_w_h1(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulh_w_h1(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhr_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhr_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhr_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhr_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhru_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhru_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhru_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhru_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhsu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhsu_w(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_h_b0(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhsu_h_b0_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_w_h0(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhsu_w_h0(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_h_b1(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhsu_h_b1_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhsu_w_h1(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhsu_w_h1(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhrsu_h(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhrsu_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pmulhrsu_w(uint64_t __x, uint64_t __y) {
  return __builtin_riscv_pmulhrsu_w(__x, __y);
}
#endif

#endif // defined(__riscv_p)

#if defined(__cplusplus)
}
#endif

#endif
