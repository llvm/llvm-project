// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -emit-llvm %s \
// RUN:            -target-feature +dspr2 -o - \
// RUN:   | FileCheck %s

typedef int q31;
typedef int i32;
typedef unsigned int ui32;
typedef long long a64;

typedef signed char v4i8 __attribute__ ((vector_size(4)));
typedef signed char v4q7 __attribute__ ((vector_size(4)));
typedef short v2i16 __attribute__ ((vector_size(4)));
typedef short v2q15 __attribute__ ((vector_size(4)));

void foo() {
  v2q15 v2q15_r, v2q15_a, v2q15_b, v2q15_c;
  v2i16 v2i16_r, v2i16_a, v2i16_b, v2i16_c;
  v4q7 v4q7_r, v4q7_a, v4q7_b;
  v4i8 v4i8_r, v4i8_a, v4i8_b, v4i8_c;
  q31 q31_r, q31_a, q31_b, q31_c;
  i32 i32_r, i32_a, i32_b, i32_c;
  ui32 ui32_r, ui32_a, ui32_b, ui32_c;
  a64 a64_r, a64_a, a64_b;

  // MIPS DSP Rev 1

  v4i8_a = (v4i8) {1, 2, 3, 0xFF};
  v4i8_b = (v4i8) {2, 4, 6, 8};
  v4i8_r = __builtin_mips_addu_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.addu.qb
  v4i8_r = __builtin_mips_addu_s_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.addu.s.qb
  v4i8_r = __builtin_mips_subu_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.subu.qb
  v4i8_r = __builtin_mips_subu_s_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.subu.s.qb

  v2q15_a = (v2q15) {0x0000, 0x8000};
  v2q15_b = (v2q15) {0x8000, 0x8000};
  v2q15_r = __builtin_mips_addq_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.addq.ph
  v2q15_r = __builtin_mips_addq_s_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.addq.s.ph
  v2q15_r = __builtin_mips_subq_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.subq.ph
  v2q15_r = __builtin_mips_subq_s_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.subq.s.ph

  a64_a = 0x12345678;
  i32_b = 0x80000000;
  i32_c = 0x11112222;
  a64_r = __builtin_mips_madd(a64_a, i32_b, i32_c);
// CHECK: call i64 @llvm.mips.madd
  a64_a = 0x12345678;
  ui32_b = 0x80000000;
  ui32_c = 0x11112222;
  a64_r = __builtin_mips_maddu(a64_a, ui32_b, ui32_c);
// CHECK: call i64 @llvm.mips.maddu
  a64_a = 0x12345678;
  i32_b = 0x80000000;
  i32_c = 0x11112222;
  a64_r = __builtin_mips_msub(a64_a, i32_b, i32_c);
// CHECK: call i64 @llvm.mips.msub
  a64_a = 0x12345678;
  ui32_b = 0x80000000;
  ui32_c = 0x11112222;
  a64_r = __builtin_mips_msubu(a64_a, ui32_b, ui32_c);
// CHECK: call i64 @llvm.mips.msubu

  q31_a = 0x12345678;
  q31_b = 0x7FFFFFFF;
  q31_r = __builtin_mips_addq_s_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.addq.s.w
  q31_r = __builtin_mips_subq_s_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.subq.s.w

  i32_a = 0xFFFFFFFF;
  i32_b = 1;
  i32_r = __builtin_mips_addsc(i32_a, i32_b);
// CHECK: call i32 @llvm.mips.addsc
  i32_a = 0;
  i32_b = 1;
  i32_r = __builtin_mips_addwc(i32_a, i32_b);
// CHECK: call i32 @llvm.mips.addwc

  i32_a = 20;
  i32_b = 0x1402;
  i32_r = __builtin_mips_modsub(i32_a, i32_b);
// CHECK: call i32 @llvm.mips.modsub

  v4i8_a = (v4i8) {1, 2, 3, 4};
  i32_r = __builtin_mips_raddu_w_qb(v4i8_a);
// CHECK: call i32 @llvm.mips.raddu.w.qb

  v2q15_a = (v2q15) {0xFFFF, 0x8000};
  v2q15_r = __builtin_mips_absq_s_ph(v2q15_a);
// CHECK: call <2 x i16> @llvm.mips.absq.s.ph
  q31_a = 0x80000000;
  q31_r = __builtin_mips_absq_s_w(q31_a);
// CHECK: call i32 @llvm.mips.absq.s.w

  v2q15_a = (v2q15) {0x1234, 0x5678};
  v2q15_b = (v2q15) {0x1111, 0x2222};
  v4i8_r = __builtin_mips_precrq_qb_ph(v2q15_a, v2q15_b);
// CHECK: call <4 x i8> @llvm.mips.precrq.qb.ph

  v2q15_a = (v2q15) {0x7F79, 0xFFFF};
  v2q15_b = (v2q15) {0x7F81, 0x2000};
  v4i8_r = __builtin_mips_precrqu_s_qb_ph(v2q15_a, v2q15_b);
// CHECK: call <4 x i8> @llvm.mips.precrqu.s.qb.ph
  q31_a = 0x12345678;
  q31_b = 0x11112222;
  v2q15_r = __builtin_mips_precrq_ph_w(q31_a, q31_b);
// CHECK: call <2 x i16> @llvm.mips.precrq.ph.w
  q31_a = 0x7000FFFF;
  q31_b = 0x80000000;
  v2q15_r = __builtin_mips_precrq_rs_ph_w(q31_a, q31_b);
// CHECK: call <2 x i16> @llvm.mips.precrq.rs.ph.w
  v2q15_a = (v2q15) {0x1234, 0x5678};
  q31_r = __builtin_mips_preceq_w_phl(v2q15_a);
// CHECK: call i32 @llvm.mips.preceq.w.phl
  q31_r = __builtin_mips_preceq_w_phr(v2q15_a);
// CHECK: call i32 @llvm.mips.preceq.w.phr
  v4i8_a = (v4i8) {0x12, 0x34, 0x56, 0x78};
  v2q15_r = __builtin_mips_precequ_ph_qbl(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.precequ.ph.qbl
  v2q15_r = __builtin_mips_precequ_ph_qbr(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.precequ.ph.qbr
  v2q15_r = __builtin_mips_precequ_ph_qbla(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.precequ.ph.qbla
  v2q15_r = __builtin_mips_precequ_ph_qbra(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.precequ.ph.qbra
  v2q15_r = __builtin_mips_preceu_ph_qbl(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.preceu.ph.qbl
  v2q15_r = __builtin_mips_preceu_ph_qbr(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.preceu.ph.qbr
  v2q15_r = __builtin_mips_preceu_ph_qbla(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.preceu.ph.qbla
  v2q15_r = __builtin_mips_preceu_ph_qbra(v4i8_a);
// CHECK: call <2 x i16> @llvm.mips.preceu.ph.qbra

  v4i8_a = (v4i8) {1, 2, 3, 4};
  v4i8_r = __builtin_mips_shll_qb(v4i8_a, 2);
// CHECK: call <4 x i8> @llvm.mips.shll.qb
  v4i8_a = (v4i8) {128, 64, 32, 16};
  v4i8_r = __builtin_mips_shrl_qb(v4i8_a, 2);
// CHECK: call <4 x i8> @llvm.mips.shrl.qb
  v2q15_a = (v2q15) {0x0001, 0x8000};
  v2q15_r = __builtin_mips_shll_ph(v2q15_a, 2);
// CHECK: call <2 x i16> @llvm.mips.shll.ph
  v2q15_r = __builtin_mips_shll_s_ph(v2q15_a, 2);
// CHECK: call <2 x i16> @llvm.mips.shll.s.ph
  v2q15_a = (v2q15) {0x7FFF, 0x8000};
  v2q15_r = __builtin_mips_shra_ph(v2q15_a, 2);
// CHECK: call <2 x i16> @llvm.mips.shra.ph
  v2q15_r = __builtin_mips_shra_r_ph(v2q15_a, 2);
// CHECK: call <2 x i16> @llvm.mips.shra.r.ph
  q31_a = 0x70000000;
  q31_r = __builtin_mips_shll_s_w(q31_a, 2);
// CHECK: call i32 @llvm.mips.shll.s.w
  q31_a = 0x7FFFFFFF;
  q31_r = __builtin_mips_shra_r_w(q31_a, 2);
// CHECK: call i32 @llvm.mips.shra.r.w
  a64_a = 0x1234567887654321LL;
  a64_r = __builtin_mips_shilo(a64_a, -8);
// CHECK: call i64 @llvm.mips.shilo

  v4i8_a = (v4i8) {0x1, 0x3, 0x5, 0x7};
  v2q15_b = (v2q15) {0x1234, 0x5678};
  v2q15_r = __builtin_mips_muleu_s_ph_qbl(v4i8_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.muleu.s.ph.qbl
  v2q15_r = __builtin_mips_muleu_s_ph_qbr(v4i8_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.muleu.s.ph.qbr
  v2q15_a = (v2q15) {0x7FFF, 0x8000};
  v2q15_b = (v2q15) {0x7FFF, 0x8000};
  v2q15_r = __builtin_mips_mulq_rs_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.mulq.rs.ph
  v2q15_a = (v2q15) {0x1234, 0x8000};
  v2q15_b = (v2q15) {0x5678, 0x8000};
  q31_r = __builtin_mips_muleq_s_w_phl(v2q15_a, v2q15_b);
// CHECK: call i32 @llvm.mips.muleq.s.w.phl
  q31_r = __builtin_mips_muleq_s_w_phr(v2q15_a, v2q15_b);
// CHECK: call i32 @llvm.mips.muleq.s.w.phr
  a64_a = 0;
  v2q15_a = (v2q15) {0x0001, 0x8000};
  v2q15_b = (v2q15) {0x0002, 0x8000};
  a64_r = __builtin_mips_mulsaq_s_w_ph(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.mulsaq.s.w.ph
  a64_a = 0;
  v2q15_b = (v2q15) {0x0001, 0x8000};
  v2q15_c = (v2q15) {0x0002, 0x8000};
  a64_r = __builtin_mips_maq_s_w_phl(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.maq.s.w.phl
  a64_r = __builtin_mips_maq_s_w_phr(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.maq.s.w.phr
  a64_a = 0x7FFFFFF0;
  a64_r = __builtin_mips_maq_sa_w_phl(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.maq.sa.w.phl
  a64_r = __builtin_mips_maq_sa_w_phr(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.maq.sa.w.phr
  i32_a = 0x80000000;
  i32_b = 0x11112222;
  a64_r = __builtin_mips_mult(i32_a, i32_b);
// CHECK: call i64 @llvm.mips.mult
  ui32_a = 0x80000000;
  ui32_b = 0x11112222;
  a64_r = __builtin_mips_multu(ui32_a, ui32_b);
// CHECK: call i64 @llvm.mips.multu

  a64_a = 0;
  v4i8_b = (v4i8) {1, 2, 3, 4};
  v4i8_c = (v4i8) {4, 5, 6, 7};
  a64_r = __builtin_mips_dpau_h_qbl(a64_a, v4i8_b, v4i8_c);
// CHECK: call i64 @llvm.mips.dpau.h.qbl
  a64_r = __builtin_mips_dpau_h_qbr(a64_a, v4i8_b, v4i8_c);
// CHECK: call i64 @llvm.mips.dpau.h.qbr
  a64_r = __builtin_mips_dpsu_h_qbl(a64_a, v4i8_b, v4i8_c);
// CHECK: call i64 @llvm.mips.dpsu.h.qbl
  a64_r = __builtin_mips_dpsu_h_qbr(a64_a, v4i8_b, v4i8_c);
// CHECK: call i64 @llvm.mips.dpsu.h.qbr
  a64_a = 0;
  v2q15_b = (v2q15) {0x0001, 0x8000};
  v2q15_c = (v2q15) {0x0002, 0x8000};
  a64_r = __builtin_mips_dpaq_s_w_ph(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.dpaq.s.w.ph
  a64_r = __builtin_mips_dpsq_s_w_ph(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.dpsq.s.w.ph
  a64_a = 0;
  q31_b = 0x80000000;
  q31_c = 0x80000000;
  a64_r = __builtin_mips_dpaq_sa_l_w(a64_a, q31_b, q31_c);
// CHECK: call i64 @llvm.mips.dpaq.sa.l.w
  a64_r = __builtin_mips_dpsq_sa_l_w(a64_a, q31_b, q31_c);
// CHECK: call i64 @llvm.mips.dpsq.sa.l.w

  v4i8_a = (v4i8) {1, 4, 10, 8};
  v4i8_b = (v4i8) {1, 2, 100, 8};
  __builtin_mips_cmpu_eq_qb(v4i8_a, v4i8_b);
// CHECK: call void @llvm.mips.cmpu.eq.qb
  __builtin_mips_cmpu_lt_qb(v4i8_a, v4i8_b);
// CHECK: call void @llvm.mips.cmpu.lt.qb
  __builtin_mips_cmpu_le_qb(v4i8_a, v4i8_b);
// CHECK: call void @llvm.mips.cmpu.le.qb
  i32_r = __builtin_mips_cmpgu_eq_qb(v4i8_a, v4i8_b);
// CHECK: call i32 @llvm.mips.cmpgu.eq.qb
  i32_r = __builtin_mips_cmpgu_lt_qb(v4i8_a, v4i8_b);
// CHECK: call i32 @llvm.mips.cmpgu.lt.qb
  i32_r = __builtin_mips_cmpgu_le_qb(v4i8_a, v4i8_b);
// CHECK: call i32 @llvm.mips.cmpgu.le.qb
  v2q15_a = (v2q15) {0x1111, 0x1234};
  v2q15_b = (v2q15) {0x4444, 0x1234};
  __builtin_mips_cmp_eq_ph(v2q15_a, v2q15_b);
// CHECK: call void @llvm.mips.cmp.eq.ph
  __builtin_mips_cmp_lt_ph(v2q15_a, v2q15_b);
// CHECK: call void @llvm.mips.cmp.lt.ph
  __builtin_mips_cmp_le_ph(v2q15_a, v2q15_b);
// CHECK: call void @llvm.mips.cmp.le.ph

  a64_a = 0xFFFFF81230000000LL;
  i32_r = __builtin_mips_extr_s_h(a64_a, 4);
// CHECK: call i32 @llvm.mips.extr.s.h
  a64_a = 0x8123456712345678LL;
  i32_r = __builtin_mips_extr_w(a64_a, 31);
// CHECK: call i32 @llvm.mips.extr.w
  i32_r = __builtin_mips_extr_rs_w(a64_a, 31);
// CHECK: call i32 @llvm.mips.extr.rs.w
  i32_r = __builtin_mips_extr_r_w(a64_a, 31);
// CHECK: call i32 @llvm.mips.extr.r.w
  a64_a = 0x1234567887654321LL;
  i32_r = __builtin_mips_extp(a64_a, 3);
// CHECK: call i32 @llvm.mips.extp
  a64_a = 0x123456789ABCDEF0LL;
  i32_r = __builtin_mips_extpdp(a64_a, 7);
// CHECK: call i32 @llvm.mips.extpdp

  __builtin_mips_wrdsp(2052, 3);
// CHECK: call void @llvm.mips.wrdsp
  i32_r = __builtin_mips_rddsp(3);
// CHECK: call i32 @llvm.mips.rddsp
  i32_a = 0xFFFFFFFF;
  i32_b = 0x12345678;
  __builtin_mips_wrdsp((16<<7) + 4, 3);
// CHECK: call void @llvm.mips.wrdsp
  i32_r = __builtin_mips_insv(i32_a, i32_b);
// CHECK: call i32 @llvm.mips.insv
  i32_a = 0x1234;
  i32_r = __builtin_mips_bitrev(i32_a);
// CHECK: call i32 @llvm.mips.bitrev
  v2q15_a = (v2q15) {0x1111, 0x2222};
  v2q15_b = (v2q15) {0x3333, 0x4444};
  v2q15_r = __builtin_mips_packrl_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.packrl.ph
  i32_a = 100;
  v4i8_r = __builtin_mips_repl_qb(i32_a);
// CHECK: call <4 x i8> @llvm.mips.repl.qb
  i32_a = 0x1234;
  v2q15_r = __builtin_mips_repl_ph(i32_a);
// CHECK: call <2 x i16> @llvm.mips.repl.ph
  v4i8_a = (v4i8) {1, 4, 10, 8};
  v4i8_b = (v4i8) {1, 2, 100, 8};
  __builtin_mips_cmpu_eq_qb(v4i8_a, v4i8_b);
// CHECK: call void @llvm.mips.cmpu.eq.qb
  v4i8_r = __builtin_mips_pick_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.pick.qb
  v2q15_a = (v2q15) {0x1111, 0x1234};
  v2q15_b = (v2q15) {0x4444, 0x1234};
  __builtin_mips_cmp_eq_ph(v2q15_a, v2q15_b);
// CHECK: call void @llvm.mips.cmp.eq.ph
  v2q15_r = __builtin_mips_pick_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.pick.ph
  a64_a = 0x1234567887654321LL;
  i32_b = 0x11112222;
  __builtin_mips_wrdsp(0, 1);
// CHECK: call void @llvm.mips.wrdsp
  a64_r = __builtin_mips_mthlip(a64_a, i32_b);
// CHECK: call i64 @llvm.mips.mthlip
  i32_r = __builtin_mips_bposge32();
// CHECK: call i32 @llvm.mips.bposge32
  char array_a[100];
  i32_r = __builtin_mips_lbux(array_a, 20);
// CHECK: call i32 @llvm.mips.lbux
  short array_b[100];
  i32_r = __builtin_mips_lhx(array_b, 20);
// CHECK: call i32 @llvm.mips.lhx
  int array_c[100];
  i32_r = __builtin_mips_lwx(array_c, 20);
// CHECK: call i32 @llvm.mips.lwx

  // MIPS DSP Rev 2

  v4q7_a = (v4q7) {0x81, 0xff, 0x80, 0x23};
  v4q7_r = __builtin_mips_absq_s_qb (v4q7_a);
// CHECK: call <4 x i8> @llvm.mips.absq.s.qb

  v2q15_a = (v2q15) {0x3334, 0x4444};
  v2q15_b = (v2q15) {0x1111, 0x2222};
  v2q15_r = __builtin_mips_addqh_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.addqh.ph
  v2q15_a = (v2q15) {0x3334, 0x4444};
  v2q15_b = (v2q15) {0x1111, 0x2222};
  v2q15_r = __builtin_mips_addqh_r_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.addqh.r.ph
  q31_a = 0x11111112;
  q31_b = 0x99999999;
  q31_r = __builtin_mips_addqh_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.addqh.w
  q31_a = 0x11111112;
  q31_b = 0x99999999;
  q31_r = __builtin_mips_addqh_r_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.addqh.r.w

  v2i16_a = (v2i16) {0xffff, 0x2468};
  v2i16_b = (v2i16) {0x1234, 0x1111};
  v2i16_r = __builtin_mips_addu_ph(v2i16_a, v2i16_b);
// CHECK: call <2 x i16> @llvm.mips.addu.ph
  v2i16_a = (v2i16) {0xffff, 0x2468};
  v2i16_b = (v2i16) {0x1234, 0x1111};
  v2i16_r = __builtin_mips_addu_s_ph(v2i16_a, v2i16_b);
// CHECK: call <2 x i16> @llvm.mips.addu.s.ph
  v4i8_a = (v4i8) {0x11, 0x22, 0x33, 0xff};
  v4i8_b = (v4i8) {0x11, 0x33, 0x99, 0xff};
  v4i8_r = __builtin_mips_adduh_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.adduh.qb
  v4i8_a = (v4i8) {0x11, 0x22, 0x33, 0xff};
  v4i8_b = (v4i8) {0x11, 0x33, 0x99, 0xff};
  v4i8_r = __builtin_mips_adduh_r_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.adduh.r.qb

  i32_a = 0x12345678;
  i32_b = 0x87654321;
  i32_r = __builtin_mips_append(i32_a, i32_b, 16);
// CHECK: call i32 @llvm.mips.append
  i32_a = 0x12345678;
  i32_b = 0x87654321;
  i32_r = __builtin_mips_balign(i32_a, i32_b, 3);
// CHECK: call i32 @llvm.mips.balign

  v4i8_a = (v4i8) {0x11, 0x22, 0x33, 0x44};
  v4i8_b = (v4i8) {0x11, 0x33, 0x33, 0x44};
  i32_r = __builtin_mips_cmpgdu_eq_qb(v4i8_a, v4i8_b);
// CHECK: call i32 @llvm.mips.cmpgdu.eq.qb
  v4i8_a = (v4i8) {0x11, 0x22, 0x33, 0x44};
  v4i8_b = (v4i8) {0x11, 0x33, 0x33, 0x44};
  i32_r = __builtin_mips_cmpgdu_lt_qb(v4i8_a, v4i8_b);
// CHECK: call i32 @llvm.mips.cmpgdu.lt.qb
  v4i8_a = (v4i8) {0x11, 0x22, 0x33, 0x54};
  v4i8_b = (v4i8) {0x11, 0x33, 0x33, 0x44};
  i32_r = __builtin_mips_cmpgdu_le_qb(v4i8_a, v4i8_b);
// CHECK: call i32 @llvm.mips.cmpgdu.le.qb

  a64_a = 0x12345678;
  v2i16_b = (v2i16) {0xffff, 0x1555};
  v2i16_c = (v2i16) {0x1234, 0x3322};
  a64_r = __builtin_mips_dpa_w_ph(a64_a, v2i16_b, v2i16_c);
// CHECK: call i64 @llvm.mips.dpa.w.ph
  a64_a = 0x12345678;
  v2i16_b = (v2i16) {0xffff, 0x1555};
  v2i16_c = (v2i16) {0x1234, 0x3322};
  a64_r = __builtin_mips_dps_w_ph(a64_a, v2i16_b, v2i16_c);
// CHECK: call i64 @llvm.mips.dps.w.ph

  a64_a = 0x70000000;
  v2q15_b = (v2q15) {0x4000, 0x2000};
  v2q15_c = (v2q15) {0x2000, 0x4000};
  a64_r = __builtin_mips_dpaqx_s_w_ph(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.dpaqx.s.w.ph
  a64_a = 0x70000000;
  v2q15_b = (v2q15) {0x4000, 0x2000};
  v2q15_c = (v2q15) {0x2000, 0x4000};
  a64_r = __builtin_mips_dpaqx_sa_w_ph(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.dpaqx.sa.w.ph
  a64_a = 0x1111222212345678LL;
  v2i16_b = (v2i16) {0x1, 0x2};
  v2i16_c = (v2i16) {0x3, 0x4};
  a64_r = __builtin_mips_dpax_w_ph(a64_a, v2i16_b, v2i16_c);
// CHECK: call i64 @llvm.mips.dpax.w.ph
  a64_a = 0x9999111112345678LL;
  v2i16_b = (v2i16) {0x1, 0x2};
  v2i16_c = (v2i16) {0x3, 0x4};
  a64_r = __builtin_mips_dpsx_w_ph(a64_a, v2i16_b, v2i16_c);
// CHECK: call i64 @llvm.mips.dpsx.w.ph
  a64_a = 0x70000000;
  v2q15_b = (v2q15) {0x4000, 0x2000};
  v2q15_c = (v2q15) {0x2000, 0x4000};
  a64_r = __builtin_mips_dpsqx_s_w_ph(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.dpsqx.s.w.ph
  a64_a = 0xFFFFFFFF80000000LL;
  v2q15_b = (v2q15) {0x4000, 0x2000};
  v2q15_c = (v2q15) {0x2000, 0x4000};
  a64_r = __builtin_mips_dpsqx_sa_w_ph(a64_a, v2q15_b, v2q15_c);
// CHECK: call i64 @llvm.mips.dpsqx.sa.w.ph

  v2i16_a = (v2i16) {0xffff, 0x2468};
  v2i16_b = (v2i16) {0x1234, 0x1111};
  v2i16_r = __builtin_mips_mul_ph(v2i16_a, v2i16_b);
// CHECK: call <2 x i16> @llvm.mips.mul.ph
  v2i16_a = (v2i16) {0x8000, 0x7fff};
  v2i16_b = (v2i16) {0x1234, 0x1111};
  v2i16_r = __builtin_mips_mul_s_ph(v2i16_a, v2i16_b);
// CHECK: call <2 x i16> @llvm.mips.mul.s.ph

  q31_a = 0x80000000;
  q31_b = 0x80000000; 
  q31_r = __builtin_mips_mulq_rs_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.mulq.rs.w
  v2q15_a = (v2q15) {0xffff, 0x8000};
  v2q15_b = (v2q15) {0x1111, 0x8000};
  v2q15_r = __builtin_mips_mulq_s_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.mulq.s.ph
  q31_a = 0x00000002;
  q31_b = 0x80000000; 
  q31_r = __builtin_mips_mulq_s_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.mulq.s.w
  a64_a = 0x19848419;
  v2i16_b = (v2i16) {0xffff, 0x8000};
  v2i16_c = (v2i16) {0x1111, 0x8000};
  a64_r = __builtin_mips_mulsa_w_ph(a64_a, v2i16_b, v2i16_c);
// CHECK: call i64 @llvm.mips.mulsa.w.ph

  v2i16_a = (v2i16) {0x1234, 0x5678};
  v2i16_b = (v2i16) {0x2233, 0x5566};
  v4i8_r = __builtin_mips_precr_qb_ph(v2i16_a, v2i16_b);
// CHECK: call <4 x i8> @llvm.mips.precr.qb.ph
  i32_a = 0x12345678;
  i32_b = 0x33334444;
  v2i16_r = __builtin_mips_precr_sra_ph_w(i32_a, i32_b, 4);
// CHECK: call <2 x i16> @llvm.mips.precr.sra.ph.w
  i32_a = 0x12345678;
  i32_b = 0x33334444;
  v2i16_r = __builtin_mips_precr_sra_r_ph_w(i32_a, i32_b, 4);
// CHECK: call <2 x i16> @llvm.mips.precr.sra.r.ph.w

  i32_a = 0x12345678;
  i32_b = 0x87654321;
  i32_r = __builtin_mips_prepend(i32_a, i32_b, 16);
// CHECK: call i32 @llvm.mips.prepend

  v4i8_a = (v4i8) {0x12, 0x45, 0x77, 0x99};
  v4i8_r = __builtin_mips_shra_qb(v4i8_a, 1);
// CHECK: call <4 x i8> @llvm.mips.shra.qb
  v4i8_a = (v4i8) {0x12, 0x45, 0x77, 0x99};
  i32_b = 1;  
  v4i8_r = __builtin_mips_shra_qb(v4i8_a, i32_b);
// CHECK: call <4 x i8> @llvm.mips.shra.qb
  v4i8_a = (v4i8) {0x12, 0x45, 0x77, 0x99};
  v4i8_r = __builtin_mips_shra_r_qb(v4i8_a, 1);
// CHECK: call <4 x i8> @llvm.mips.shra.r.qb
  v4i8_a = (v4i8) {0x12, 0x45, 0x77, 0x99};
  i32_b = 1;  
  v4i8_r = __builtin_mips_shra_r_qb(v4i8_a, i32_b);
// CHECK: call <4 x i8> @llvm.mips.shra.r.qb
  v2i16_a = (v2i16) {0x1357, 0x2468};
  v2i16_r = __builtin_mips_shrl_ph(v2i16_a, 4);
// CHECK: call <2 x i16> @llvm.mips.shrl.ph
  v2i16_a = (v2i16) {0x1357, 0x2468};
  i32_b = 8;
  v2i16_r = __builtin_mips_shrl_ph (v2i16_a, i32_b);
// CHECK: call <2 x i16> @llvm.mips.shrl.ph

  v2q15_a = (v2q15) {0x3334, 0x4444};
  v2q15_b = (v2q15) {0x1111, 0x2222};
  v2q15_r = __builtin_mips_subqh_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.subqh.ph
  v2q15_a = (v2q15) {0x3334, 0x4444};
  v2q15_b = (v2q15) {0x1111, 0x2222};
  v2q15_r = __builtin_mips_subqh_r_ph(v2q15_a, v2q15_b);
// CHECK: call <2 x i16> @llvm.mips.subqh.r.ph
  q31_a = 0x11111112;
  q31_b = 0x99999999;
  q31_r = __builtin_mips_subqh_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.subqh.w
  q31_a = 0x11111112;
  q31_b = 0x99999999;
  q31_r = __builtin_mips_subqh_r_w(q31_a, q31_b);
// CHECK: call i32 @llvm.mips.subqh.r.w

  v2i16_a = (v2i16) {0x1357, 0x4455};
  v2i16_b = (v2i16) {0x3333, 0x4444};
  v2i16_r = __builtin_mips_subu_ph(v2i16_a, v2i16_b);
// CHECK: call <2 x i16> @llvm.mips.subu.ph
  v2i16_a = (v2i16) {0x1357, 0x4455};
  v2i16_b = (v2i16) {0x3333, 0x4444};
  v2i16_r = __builtin_mips_subu_s_ph(v2i16_a, v2i16_b);
// CHECK: call <2 x i16> @llvm.mips.subu.s.ph

  v4i8_a = (v4i8) {0x33 ,0x44, 0x55, 0x66};
  v4i8_b = (v4i8) {0x99 ,0x15, 0x85, 0xff};
  v4i8_r = __builtin_mips_subuh_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.subuh.qb
  v4i8_a = (v4i8) {0x33 ,0x44, 0x55, 0x66};
  v4i8_b = (v4i8) {0x99 ,0x15, 0x85, 0xff};
  v4i8_r = __builtin_mips_subuh_r_qb(v4i8_a, v4i8_b);
// CHECK: call <4 x i8> @llvm.mips.subuh.r.qb
}

void test_eh_return_data_regno()
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK: store volatile i32 4
  res = __builtin_eh_return_data_regno(1);  // CHECK: store volatile i32 5
}
