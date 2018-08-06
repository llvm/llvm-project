// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv65 -target-feature +hvxv65 -target-feature +hvx-length64b -emit-llvm %s -o - | FileCheck %s

void test() {
  int v64 __attribute__((__vector_size__(64)));
  int v128 __attribute__((__vector_size__(128)));

  // CHECK: @llvm.hexagon.V6.extractw
  __builtin_HEXAGON_V6_extractw(v64, 0);
  // CHECK: @llvm.hexagon.V6.hi
  __builtin_HEXAGON_V6_hi(v128);
  // CHECK: @llvm.hexagon.V6.lo
  __builtin_HEXAGON_V6_lo(v128);
  // CHECK: @llvm.hexagon.V6.lvsplatb
  __builtin_HEXAGON_V6_lvsplatb(0);
  // CHECK: @llvm.hexagon.V6.lvsplath
  __builtin_HEXAGON_V6_lvsplath(0);
  // CHECK: @llvm.hexagon.V6.lvsplatw
  __builtin_HEXAGON_V6_lvsplatw(0);
  // CHECK: @llvm.hexagon.V6.pred.and
  __builtin_HEXAGON_V6_pred_and(v64, v64);
  // CHECK: @llvm.hexagon.V6.pred.and.n
  __builtin_HEXAGON_V6_pred_and_n(v64, v64);
  // CHECK: @llvm.hexagon.V6.pred.not
  __builtin_HEXAGON_V6_pred_not(v64);
  // CHECK: @llvm.hexagon.V6.pred.or
  __builtin_HEXAGON_V6_pred_or(v64, v64);
  // CHECK: @llvm.hexagon.V6.pred.or.n
  __builtin_HEXAGON_V6_pred_or_n(v64, v64);
  // CHECK: @llvm.hexagon.V6.pred.scalar2
  __builtin_HEXAGON_V6_pred_scalar2(0);
  // CHECK: @llvm.hexagon.V6.pred.scalar2v2
  __builtin_HEXAGON_V6_pred_scalar2v2(0);
  // CHECK: @llvm.hexagon.V6.pred.xor
  __builtin_HEXAGON_V6_pred_xor(v64, v64);
  // CHECK: @llvm.hexagon.V6.shuffeqh
  __builtin_HEXAGON_V6_shuffeqh(v64, v64);
  // CHECK: @llvm.hexagon.V6.shuffeqw
  __builtin_HEXAGON_V6_shuffeqw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vS32b.nqpred.ai
  __builtin_HEXAGON_V6_vS32b_nqpred_ai(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vS32b.nt.nqpred.ai
  __builtin_HEXAGON_V6_vS32b_nt_nqpred_ai(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vS32b.nt.qpred.ai
  __builtin_HEXAGON_V6_vS32b_nt_qpred_ai(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vS32b.qpred.ai
  __builtin_HEXAGON_V6_vS32b_qpred_ai(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vabsb
  __builtin_HEXAGON_V6_vabsb(v64);
  // CHECK: @llvm.hexagon.V6.vabsb.sat
  __builtin_HEXAGON_V6_vabsb_sat(v64);
  // CHECK: @llvm.hexagon.V6.vabsdiffh
  __builtin_HEXAGON_V6_vabsdiffh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vabsdiffub
  __builtin_HEXAGON_V6_vabsdiffub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vabsdiffuh
  __builtin_HEXAGON_V6_vabsdiffuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vabsdiffw
  __builtin_HEXAGON_V6_vabsdiffw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vabsh
  __builtin_HEXAGON_V6_vabsh(v64);
  // CHECK: @llvm.hexagon.V6.vabsh.sat
  __builtin_HEXAGON_V6_vabsh_sat(v64);
  // CHECK: @llvm.hexagon.V6.vabsw
  __builtin_HEXAGON_V6_vabsw(v64);
  // CHECK: @llvm.hexagon.V6.vabsw.sat
  __builtin_HEXAGON_V6_vabsw_sat(v64);
  // CHECK: @llvm.hexagon.V6.vaddb
  __builtin_HEXAGON_V6_vaddb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddb.dv
  __builtin_HEXAGON_V6_vaddb_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddbnq
  __builtin_HEXAGON_V6_vaddbnq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddbq
  __builtin_HEXAGON_V6_vaddbq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddbsat
  __builtin_HEXAGON_V6_vaddbsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddbsat.dv
  __builtin_HEXAGON_V6_vaddbsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddcarry
  __builtin_HEXAGON_V6_vaddcarry(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vaddclbh
  __builtin_HEXAGON_V6_vaddclbh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddclbw
  __builtin_HEXAGON_V6_vaddclbw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddh
  __builtin_HEXAGON_V6_vaddh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddh.dv
  __builtin_HEXAGON_V6_vaddh_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddhnq
  __builtin_HEXAGON_V6_vaddhnq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddhq
  __builtin_HEXAGON_V6_vaddhq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddhsat
  __builtin_HEXAGON_V6_vaddhsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddhsat.dv
  __builtin_HEXAGON_V6_vaddhsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddhw
  __builtin_HEXAGON_V6_vaddhw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddhw.acc
  __builtin_HEXAGON_V6_vaddhw_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddubh
  __builtin_HEXAGON_V6_vaddubh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddubh.acc
  __builtin_HEXAGON_V6_vaddubh_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddubsat
  __builtin_HEXAGON_V6_vaddubsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddubsat.dv
  __builtin_HEXAGON_V6_vaddubsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddububb.sat
  __builtin_HEXAGON_V6_vaddububb_sat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vadduhsat
  __builtin_HEXAGON_V6_vadduhsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vadduhsat.dv
  __builtin_HEXAGON_V6_vadduhsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vadduhw
  __builtin_HEXAGON_V6_vadduhw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vadduhw.acc
  __builtin_HEXAGON_V6_vadduhw_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vadduwsat
  __builtin_HEXAGON_V6_vadduwsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vadduwsat.dv
  __builtin_HEXAGON_V6_vadduwsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddw
  __builtin_HEXAGON_V6_vaddw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddw.dv
  __builtin_HEXAGON_V6_vaddw_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddwnq
  __builtin_HEXAGON_V6_vaddwnq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddwq
  __builtin_HEXAGON_V6_vaddwq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddwsat
  __builtin_HEXAGON_V6_vaddwsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddwsat.dv
  __builtin_HEXAGON_V6_vaddwsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.valignb
  __builtin_HEXAGON_V6_valignb(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.valignbi
  __builtin_HEXAGON_V6_valignbi(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vand
  __builtin_HEXAGON_V6_vand(v64, v64);
  // CHECK: @llvm.hexagon.V6.vandnqrt
  __builtin_HEXAGON_V6_vandnqrt(v64, 0);
  // CHECK: @llvm.hexagon.V6.vandnqrt.acc
  __builtin_HEXAGON_V6_vandnqrt_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt
  __builtin_HEXAGON_V6_vandqrt(v64, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt.acc
  __builtin_HEXAGON_V6_vandqrt_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vandvnqv
  __builtin_HEXAGON_V6_vandvnqv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vandvqv
  __builtin_HEXAGON_V6_vandvqv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vandvrt
  __builtin_HEXAGON_V6_vandvrt(v64, 0);
  // CHECK: @llvm.hexagon.V6.vandvrt.acc
  __builtin_HEXAGON_V6_vandvrt_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vaslh
  __builtin_HEXAGON_V6_vaslh(v64, 0);
  // CHECK: @llvm.hexagon.V6.vaslh.acc
  __builtin_HEXAGON_V6_vaslh_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vaslhv
  __builtin_HEXAGON_V6_vaslhv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaslw
  __builtin_HEXAGON_V6_vaslw(v64, 0);
  // CHECK: @llvm.hexagon.V6.vaslw.acc
  __builtin_HEXAGON_V6_vaslw_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vaslwv
  __builtin_HEXAGON_V6_vaslwv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vasrh
  __builtin_HEXAGON_V6_vasrh(v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrh.acc
  __builtin_HEXAGON_V6_vasrh_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrhbrndsat
  __builtin_HEXAGON_V6_vasrhbrndsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrhbsat
  __builtin_HEXAGON_V6_vasrhbsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubrndsat
  __builtin_HEXAGON_V6_vasrhubrndsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubsat
  __builtin_HEXAGON_V6_vasrhubsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrhv
  __builtin_HEXAGON_V6_vasrhv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vasruhubrndsat
  __builtin_HEXAGON_V6_vasruhubrndsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasruhubsat
  __builtin_HEXAGON_V6_vasruhubsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasruwuhrndsat
  __builtin_HEXAGON_V6_vasruwuhrndsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasruwuhsat
  __builtin_HEXAGON_V6_vasruwuhsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrw
  __builtin_HEXAGON_V6_vasrw(v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrw.acc
  __builtin_HEXAGON_V6_vasrw_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrwh
  __builtin_HEXAGON_V6_vasrwh(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhrndsat
  __builtin_HEXAGON_V6_vasrwhrndsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhsat
  __builtin_HEXAGON_V6_vasrwhsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrwuhrndsat
  __builtin_HEXAGON_V6_vasrwuhrndsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrwuhsat
  __builtin_HEXAGON_V6_vasrwuhsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vasrwv
  __builtin_HEXAGON_V6_vasrwv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vassign
  __builtin_HEXAGON_V6_vassign(v64);
  // CHECK: @llvm.hexagon.V6.vassignp
  __builtin_HEXAGON_V6_vassignp(v128);
  // CHECK: @llvm.hexagon.V6.vavgb
  __builtin_HEXAGON_V6_vavgb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavgbrnd
  __builtin_HEXAGON_V6_vavgbrnd(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavgh
  __builtin_HEXAGON_V6_vavgh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavghrnd
  __builtin_HEXAGON_V6_vavghrnd(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavgub
  __builtin_HEXAGON_V6_vavgub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavgubrnd
  __builtin_HEXAGON_V6_vavgubrnd(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavguh
  __builtin_HEXAGON_V6_vavguh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavguhrnd
  __builtin_HEXAGON_V6_vavguhrnd(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavguw
  __builtin_HEXAGON_V6_vavguw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavguwrnd
  __builtin_HEXAGON_V6_vavguwrnd(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavgw
  __builtin_HEXAGON_V6_vavgw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vavgwrnd
  __builtin_HEXAGON_V6_vavgwrnd(v64, v64);
  // CHECK: @llvm.hexagon.V6.vcl0h
  __builtin_HEXAGON_V6_vcl0h(v64);
  // CHECK: @llvm.hexagon.V6.vcl0w
  __builtin_HEXAGON_V6_vcl0w(v64);
  // CHECK: @llvm.hexagon.V6.vcombine
  __builtin_HEXAGON_V6_vcombine(v64, v64);
  // CHECK: @llvm.hexagon.V6.vd0
  __builtin_HEXAGON_V6_vd0();
  // CHECK: @llvm.hexagon.V6.vdd0
  __builtin_HEXAGON_V6_vdd0();
  // CHECK: @llvm.hexagon.V6.vdealb
  __builtin_HEXAGON_V6_vdealb(v64);
  // CHECK: @llvm.hexagon.V6.vdealb4w
  __builtin_HEXAGON_V6_vdealb4w(v64, v64);
  // CHECK: @llvm.hexagon.V6.vdealh
  __builtin_HEXAGON_V6_vdealh(v64);
  // CHECK: @llvm.hexagon.V6.vdealvdd
  __builtin_HEXAGON_V6_vdealvdd(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdelta
  __builtin_HEXAGON_V6_vdelta(v64, v64);
  // CHECK: @llvm.hexagon.V6.vdmpybus
  __builtin_HEXAGON_V6_vdmpybus(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.acc
  __builtin_HEXAGON_V6_vdmpybus_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv
  __builtin_HEXAGON_V6_vdmpybus_dv(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv.acc
  __builtin_HEXAGON_V6_vdmpybus_dv_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb
  __builtin_HEXAGON_V6_vdmpyhb(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.acc
  __builtin_HEXAGON_V6_vdmpyhb_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv
  __builtin_HEXAGON_V6_vdmpyhb_dv(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv.acc
  __builtin_HEXAGON_V6_vdmpyhb_dv_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat
  __builtin_HEXAGON_V6_vdmpyhisat(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat.acc
  __builtin_HEXAGON_V6_vdmpyhisat_acc(v64, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat
  __builtin_HEXAGON_V6_vdmpyhsat(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat.acc
  __builtin_HEXAGON_V6_vdmpyhsat_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat
  __builtin_HEXAGON_V6_vdmpyhsuisat(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat.acc
  __builtin_HEXAGON_V6_vdmpyhsuisat_acc(v64, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat
  __builtin_HEXAGON_V6_vdmpyhsusat(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat.acc
  __builtin_HEXAGON_V6_vdmpyhsusat_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat
  __builtin_HEXAGON_V6_vdmpyhvsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat.acc
  __builtin_HEXAGON_V6_vdmpyhvsat_acc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vdsaduh
  __builtin_HEXAGON_V6_vdsaduh(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdsaduh.acc
  __builtin_HEXAGON_V6_vdsaduh_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.veqb
  __builtin_HEXAGON_V6_veqb(v64, v64);
  // CHECK: @llvm.hexagon.V6.veqb.and
  __builtin_HEXAGON_V6_veqb_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqb.or
  __builtin_HEXAGON_V6_veqb_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqb.xor
  __builtin_HEXAGON_V6_veqb_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqh
  __builtin_HEXAGON_V6_veqh(v64, v64);
  // CHECK: @llvm.hexagon.V6.veqh.and
  __builtin_HEXAGON_V6_veqh_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqh.or
  __builtin_HEXAGON_V6_veqh_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqh.xor
  __builtin_HEXAGON_V6_veqh_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqw
  __builtin_HEXAGON_V6_veqw(v64, v64);
  // CHECK: @llvm.hexagon.V6.veqw.and
  __builtin_HEXAGON_V6_veqw_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqw.or
  __builtin_HEXAGON_V6_veqw_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.veqw.xor
  __builtin_HEXAGON_V6_veqw_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgathermh
  __builtin_HEXAGON_V6_vgathermh(0, 0, 0, v64);
  // CHECK: @llvm.hexagon.V6.vgathermhq
  __builtin_HEXAGON_V6_vgathermhq(0, v64, 0, 0, v64);
  // CHECK: @llvm.hexagon.V6.vgathermhw
  __builtin_HEXAGON_V6_vgathermhw(0, 0, 0, v128);
  // CHECK: @llvm.hexagon.V6.vgathermhwq
  __builtin_HEXAGON_V6_vgathermhwq(0, v64, 0, 0, v128);
  // CHECK: @llvm.hexagon.V6.vgathermw
  __builtin_HEXAGON_V6_vgathermw(0, 0, 0, v64);
  // CHECK: @llvm.hexagon.V6.vgathermwq
  __builtin_HEXAGON_V6_vgathermwq(0, v64, 0, 0, v64);
  // CHECK: @llvm.hexagon.V6.vgtb
  __builtin_HEXAGON_V6_vgtb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtb.and
  __builtin_HEXAGON_V6_vgtb_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtb.or
  __builtin_HEXAGON_V6_vgtb_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtb.xor
  __builtin_HEXAGON_V6_vgtb_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgth
  __builtin_HEXAGON_V6_vgth(v64, v64);
  // CHECK: @llvm.hexagon.V6.vgth.and
  __builtin_HEXAGON_V6_vgth_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgth.or
  __builtin_HEXAGON_V6_vgth_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgth.xor
  __builtin_HEXAGON_V6_vgth_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtub
  __builtin_HEXAGON_V6_vgtub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtub.and
  __builtin_HEXAGON_V6_vgtub_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtub.or
  __builtin_HEXAGON_V6_vgtub_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtub.xor
  __builtin_HEXAGON_V6_vgtub_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuh
  __builtin_HEXAGON_V6_vgtuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuh.and
  __builtin_HEXAGON_V6_vgtuh_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuh.or
  __builtin_HEXAGON_V6_vgtuh_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuh.xor
  __builtin_HEXAGON_V6_vgtuh_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuw
  __builtin_HEXAGON_V6_vgtuw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuw.and
  __builtin_HEXAGON_V6_vgtuw_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuw.or
  __builtin_HEXAGON_V6_vgtuw_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtuw.xor
  __builtin_HEXAGON_V6_vgtuw_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtw
  __builtin_HEXAGON_V6_vgtw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtw.and
  __builtin_HEXAGON_V6_vgtw_and(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtw.or
  __builtin_HEXAGON_V6_vgtw_or(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vgtw.xor
  __builtin_HEXAGON_V6_vgtw_xor(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vinsertwr
  __builtin_HEXAGON_V6_vinsertwr(v64, 0);
  // CHECK: @llvm.hexagon.V6.vlalignb
  __builtin_HEXAGON_V6_vlalignb(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlalignbi
  __builtin_HEXAGON_V6_vlalignbi(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlsrb
  __builtin_HEXAGON_V6_vlsrb(v64, 0);
  // CHECK: @llvm.hexagon.V6.vlsrh
  __builtin_HEXAGON_V6_vlsrh(v64, 0);
  // CHECK: @llvm.hexagon.V6.vlsrhv
  __builtin_HEXAGON_V6_vlsrhv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vlsrw
  __builtin_HEXAGON_V6_vlsrw(v64, 0);
  // CHECK: @llvm.hexagon.V6.vlsrwv
  __builtin_HEXAGON_V6_vlsrwv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vlut4
  __builtin_HEXAGON_V6_vlut4(v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb
  __builtin_HEXAGON_V6_vlutvvb(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.nm
  __builtin_HEXAGON_V6_vlutvvb_nm(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.oracc
  __builtin_HEXAGON_V6_vlutvvb_oracc(v64, v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.oracci
  __builtin_HEXAGON_V6_vlutvvb_oracci(v64, v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvbi
  __builtin_HEXAGON_V6_vlutvvbi(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh
  __builtin_HEXAGON_V6_vlutvwh(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.nm
  __builtin_HEXAGON_V6_vlutvwh_nm(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.oracc
  __builtin_HEXAGON_V6_vlutvwh_oracc(v128, v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.oracci
  __builtin_HEXAGON_V6_vlutvwh_oracci(v128, v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwhi
  __builtin_HEXAGON_V6_vlutvwhi(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmaskedstorenq
  __builtin_HEXAGON_V6_vmaskedstorenq(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vmaskedstorentnq
  __builtin_HEXAGON_V6_vmaskedstorentnq(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vmaskedstorentq
  __builtin_HEXAGON_V6_vmaskedstorentq(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vmaskedstoreq
  __builtin_HEXAGON_V6_vmaskedstoreq(v64, 0, v64);
  // CHECK: @llvm.hexagon.V6.vmaxb
  __builtin_HEXAGON_V6_vmaxb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmaxh
  __builtin_HEXAGON_V6_vmaxh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmaxub
  __builtin_HEXAGON_V6_vmaxub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmaxuh
  __builtin_HEXAGON_V6_vmaxuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmaxw
  __builtin_HEXAGON_V6_vmaxw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vminb
  __builtin_HEXAGON_V6_vminb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vminh
  __builtin_HEXAGON_V6_vminh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vminub
  __builtin_HEXAGON_V6_vminub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vminuh
  __builtin_HEXAGON_V6_vminuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vminw
  __builtin_HEXAGON_V6_vminw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpabus
  __builtin_HEXAGON_V6_vmpabus(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpabus.acc
  __builtin_HEXAGON_V6_vmpabus_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpabusv
  __builtin_HEXAGON_V6_vmpabusv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpabuu
  __builtin_HEXAGON_V6_vmpabuu(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpabuu.acc
  __builtin_HEXAGON_V6_vmpabuu_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpabuuv
  __builtin_HEXAGON_V6_vmpabuuv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpahb
  __builtin_HEXAGON_V6_vmpahb(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpahb.acc
  __builtin_HEXAGON_V6_vmpahb_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpahhsat
  __builtin_HEXAGON_V6_vmpahhsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpauhb
  __builtin_HEXAGON_V6_vmpauhb(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpauhb.acc
  __builtin_HEXAGON_V6_vmpauhb_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpauhuhsat
  __builtin_HEXAGON_V6_vmpauhuhsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpsuhuhsat
  __builtin_HEXAGON_V6_vmpsuhuhsat(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus
  __builtin_HEXAGON_V6_vmpybus(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus.acc
  __builtin_HEXAGON_V6_vmpybus_acc(v128, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpybusv
  __builtin_HEXAGON_V6_vmpybusv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpybusv.acc
  __builtin_HEXAGON_V6_vmpybusv_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpybv
  __builtin_HEXAGON_V6_vmpybv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpybv.acc
  __builtin_HEXAGON_V6_vmpybv_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyewuh
  __builtin_HEXAGON_V6_vmpyewuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyewuh.64
  __builtin_HEXAGON_V6_vmpyewuh_64(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyh
  __builtin_HEXAGON_V6_vmpyh(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyh.acc
  __builtin_HEXAGON_V6_vmpyh_acc(v128, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsat.acc
  __builtin_HEXAGON_V6_vmpyhsat_acc(v128, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsrs
  __builtin_HEXAGON_V6_vmpyhsrs(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhss
  __builtin_HEXAGON_V6_vmpyhss(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhus
  __builtin_HEXAGON_V6_vmpyhus(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyhus.acc
  __builtin_HEXAGON_V6_vmpyhus_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyhv
  __builtin_HEXAGON_V6_vmpyhv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyhv.acc
  __builtin_HEXAGON_V6_vmpyhv_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyhvsrs
  __builtin_HEXAGON_V6_vmpyhvsrs(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyieoh
  __builtin_HEXAGON_V6_vmpyieoh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyiewh.acc
  __builtin_HEXAGON_V6_vmpyiewh_acc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh
  __builtin_HEXAGON_V6_vmpyiewuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh.acc
  __builtin_HEXAGON_V6_vmpyiewuh_acc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyih
  __builtin_HEXAGON_V6_vmpyih(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyih.acc
  __builtin_HEXAGON_V6_vmpyih_acc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyihb
  __builtin_HEXAGON_V6_vmpyihb(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyihb.acc
  __builtin_HEXAGON_V6_vmpyihb_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiowh
  __builtin_HEXAGON_V6_vmpyiowh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyiwb
  __builtin_HEXAGON_V6_vmpyiwb(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwb.acc
  __builtin_HEXAGON_V6_vmpyiwb_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh
  __builtin_HEXAGON_V6_vmpyiwh(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh.acc
  __builtin_HEXAGON_V6_vmpyiwh_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwub
  __builtin_HEXAGON_V6_vmpyiwub(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwub.acc
  __builtin_HEXAGON_V6_vmpyiwub_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyowh
  __builtin_HEXAGON_V6_vmpyowh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyowh.64.acc
  __builtin_HEXAGON_V6_vmpyowh_64_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd
  __builtin_HEXAGON_V6_vmpyowh_rnd(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd.sacc
  __builtin_HEXAGON_V6_vmpyowh_rnd_sacc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyowh.sacc
  __builtin_HEXAGON_V6_vmpyowh_sacc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyub
  __builtin_HEXAGON_V6_vmpyub(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyub.acc
  __builtin_HEXAGON_V6_vmpyub_acc(v128, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyubv
  __builtin_HEXAGON_V6_vmpyubv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyubv.acc
  __builtin_HEXAGON_V6_vmpyubv_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyuh
  __builtin_HEXAGON_V6_vmpyuh(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuh.acc
  __builtin_HEXAGON_V6_vmpyuh_acc(v128, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuhe
  __builtin_HEXAGON_V6_vmpyuhe(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuhe.acc
  __builtin_HEXAGON_V6_vmpyuhe_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuhv
  __builtin_HEXAGON_V6_vmpyuhv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpyuhv.acc
  __builtin_HEXAGON_V6_vmpyuhv_acc(v128, v64, v64);
  // CHECK: @llvm.hexagon.V6.vmux
  __builtin_HEXAGON_V6_vmux(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vnavgb
  __builtin_HEXAGON_V6_vnavgb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vnavgh
  __builtin_HEXAGON_V6_vnavgh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vnavgub
  __builtin_HEXAGON_V6_vnavgub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vnavgw
  __builtin_HEXAGON_V6_vnavgw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vnormamth
  __builtin_HEXAGON_V6_vnormamth(v64);
  // CHECK: @llvm.hexagon.V6.vnormamtw
  __builtin_HEXAGON_V6_vnormamtw(v64);
  // CHECK: @llvm.hexagon.V6.vnot
  __builtin_HEXAGON_V6_vnot(v64);
  // CHECK: @llvm.hexagon.V6.vor
  __builtin_HEXAGON_V6_vor(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackeb
  __builtin_HEXAGON_V6_vpackeb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackeh
  __builtin_HEXAGON_V6_vpackeh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackhb.sat
  __builtin_HEXAGON_V6_vpackhb_sat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackhub.sat
  __builtin_HEXAGON_V6_vpackhub_sat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackob
  __builtin_HEXAGON_V6_vpackob(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackoh
  __builtin_HEXAGON_V6_vpackoh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackwh.sat
  __builtin_HEXAGON_V6_vpackwh_sat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpackwuh.sat
  __builtin_HEXAGON_V6_vpackwuh_sat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vpopcounth
  __builtin_HEXAGON_V6_vpopcounth(v64);
  // CHECK: @llvm.hexagon.V6.vprefixqb
  __builtin_HEXAGON_V6_vprefixqb(v64);
  // CHECK: @llvm.hexagon.V6.vprefixqh
  __builtin_HEXAGON_V6_vprefixqh(v64);
  // CHECK: @llvm.hexagon.V6.vprefixqw
  __builtin_HEXAGON_V6_vprefixqw(v64);
  // CHECK: @llvm.hexagon.V6.vrdelta
  __builtin_HEXAGON_V6_vrdelta(v64, v64);
  // CHECK: @llvm.hexagon.V6.vrmpybub.rtt
  __builtin_HEXAGON_V6_vrmpybub_rtt(v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybub.rtt.acc
  __builtin_HEXAGON_V6_vrmpybub_rtt_acc(v128, v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus
  __builtin_HEXAGON_V6_vrmpybus(v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus.acc
  __builtin_HEXAGON_V6_vrmpybus_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi
  __builtin_HEXAGON_V6_vrmpybusi(v128, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi.acc
  __builtin_HEXAGON_V6_vrmpybusi_acc(v128, v128, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusv
  __builtin_HEXAGON_V6_vrmpybusv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vrmpybusv.acc
  __builtin_HEXAGON_V6_vrmpybusv_acc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vrmpybv
  __builtin_HEXAGON_V6_vrmpybv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vrmpybv.acc
  __builtin_HEXAGON_V6_vrmpybv_acc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vrmpyub
  __builtin_HEXAGON_V6_vrmpyub(v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.acc
  __builtin_HEXAGON_V6_vrmpyub_acc(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.rtt
  __builtin_HEXAGON_V6_vrmpyub_rtt(v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.rtt.acc
  __builtin_HEXAGON_V6_vrmpyub_rtt_acc(v128, v64, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi
  __builtin_HEXAGON_V6_vrmpyubi(v128, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi.acc
  __builtin_HEXAGON_V6_vrmpyubi_acc(v128, v128, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubv
  __builtin_HEXAGON_V6_vrmpyubv(v64, v64);
  // CHECK: @llvm.hexagon.V6.vrmpyubv.acc
  __builtin_HEXAGON_V6_vrmpyubv_acc(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vror
  __builtin_HEXAGON_V6_vror(v64, 0);
  // CHECK: @llvm.hexagon.V6.vroundhb
  __builtin_HEXAGON_V6_vroundhb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vroundhub
  __builtin_HEXAGON_V6_vroundhub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vrounduhub
  __builtin_HEXAGON_V6_vrounduhub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vrounduwuh
  __builtin_HEXAGON_V6_vrounduwuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vroundwh
  __builtin_HEXAGON_V6_vroundwh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vroundwuh
  __builtin_HEXAGON_V6_vroundwuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vrsadubi
  __builtin_HEXAGON_V6_vrsadubi(v128, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrsadubi.acc
  __builtin_HEXAGON_V6_vrsadubi_acc(v128, v128, 0, 0);
  // CHECK: @llvm.hexagon.V6.vsathub
  __builtin_HEXAGON_V6_vsathub(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsatuwuh
  __builtin_HEXAGON_V6_vsatuwuh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsatwh
  __builtin_HEXAGON_V6_vsatwh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsb
  __builtin_HEXAGON_V6_vsb(v64);
  // CHECK: @llvm.hexagon.V6.vscattermh
  __builtin_HEXAGON_V6_vscattermh(0, 0, v64, v64);
  // CHECK: @llvm.hexagon.V6.vscattermh.add
  __builtin_HEXAGON_V6_vscattermh_add(0, 0, v64, v64);
  // CHECK: @llvm.hexagon.V6.vscattermhq
  __builtin_HEXAGON_V6_vscattermhq(v64, 0, 0, v64, v64);
  // CHECK: @llvm.hexagon.V6.vscattermhw
  __builtin_HEXAGON_V6_vscattermhw(0, 0, v128, v64);
  // CHECK: @llvm.hexagon.V6.vscattermhw.add
  __builtin_HEXAGON_V6_vscattermhw_add(0, 0, v128, v64);
  // CHECK: @llvm.hexagon.V6.vscattermhwq
  __builtin_HEXAGON_V6_vscattermhwq(v64, 0, 0, v128, v64);
  // CHECK: @llvm.hexagon.V6.vscattermw
  __builtin_HEXAGON_V6_vscattermw(0, 0, v64, v64);
  // CHECK: @llvm.hexagon.V6.vscattermw.add
  __builtin_HEXAGON_V6_vscattermw_add(0, 0, v64, v64);
  // CHECK: @llvm.hexagon.V6.vscattermwq
  __builtin_HEXAGON_V6_vscattermwq(v64, 0, 0, v64, v64);
  // CHECK: @llvm.hexagon.V6.vsh
  __builtin_HEXAGON_V6_vsh(v64);
  // CHECK: @llvm.hexagon.V6.vshufeh
  __builtin_HEXAGON_V6_vshufeh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vshuffb
  __builtin_HEXAGON_V6_vshuffb(v64);
  // CHECK: @llvm.hexagon.V6.vshuffeb
  __builtin_HEXAGON_V6_vshuffeb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vshuffh
  __builtin_HEXAGON_V6_vshuffh(v64);
  // CHECK: @llvm.hexagon.V6.vshuffob
  __builtin_HEXAGON_V6_vshuffob(v64, v64);
  // CHECK: @llvm.hexagon.V6.vshuffvdd
  __builtin_HEXAGON_V6_vshuffvdd(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vshufoeb
  __builtin_HEXAGON_V6_vshufoeb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vshufoeh
  __builtin_HEXAGON_V6_vshufoeh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vshufoh
  __builtin_HEXAGON_V6_vshufoh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubb
  __builtin_HEXAGON_V6_vsubb(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubb.dv
  __builtin_HEXAGON_V6_vsubb_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubbnq
  __builtin_HEXAGON_V6_vsubbnq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubbq
  __builtin_HEXAGON_V6_vsubbq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubbsat
  __builtin_HEXAGON_V6_vsubbsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubbsat.dv
  __builtin_HEXAGON_V6_vsubbsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubcarry
  __builtin_HEXAGON_V6_vsubcarry(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vsubh
  __builtin_HEXAGON_V6_vsubh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubh.dv
  __builtin_HEXAGON_V6_vsubh_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubhnq
  __builtin_HEXAGON_V6_vsubhnq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubhq
  __builtin_HEXAGON_V6_vsubhq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubhsat
  __builtin_HEXAGON_V6_vsubhsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubhsat.dv
  __builtin_HEXAGON_V6_vsubhsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubhw
  __builtin_HEXAGON_V6_vsubhw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsububh
  __builtin_HEXAGON_V6_vsububh(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsububsat
  __builtin_HEXAGON_V6_vsububsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsububsat.dv
  __builtin_HEXAGON_V6_vsububsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubububb.sat
  __builtin_HEXAGON_V6_vsubububb_sat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubuhsat
  __builtin_HEXAGON_V6_vsubuhsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubuhsat.dv
  __builtin_HEXAGON_V6_vsubuhsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubuhw
  __builtin_HEXAGON_V6_vsubuhw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubuwsat
  __builtin_HEXAGON_V6_vsubuwsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubuwsat.dv
  __builtin_HEXAGON_V6_vsubuwsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubw
  __builtin_HEXAGON_V6_vsubw(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubw.dv
  __builtin_HEXAGON_V6_vsubw_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubwnq
  __builtin_HEXAGON_V6_vsubwnq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubwq
  __builtin_HEXAGON_V6_vsubwq(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubwsat
  __builtin_HEXAGON_V6_vsubwsat(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubwsat.dv
  __builtin_HEXAGON_V6_vsubwsat_dv(v128, v128);
  // CHECK: @llvm.hexagon.V6.vswap
  __builtin_HEXAGON_V6_vswap(v64, v64, v64);
  // CHECK: @llvm.hexagon.V6.vtmpyb
  __builtin_HEXAGON_V6_vtmpyb(v128, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyb.acc
  __builtin_HEXAGON_V6_vtmpyb_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus
  __builtin_HEXAGON_V6_vtmpybus(v128, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus.acc
  __builtin_HEXAGON_V6_vtmpybus_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb
  __builtin_HEXAGON_V6_vtmpyhb(v128, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb.acc
  __builtin_HEXAGON_V6_vtmpyhb_acc(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vunpackb
  __builtin_HEXAGON_V6_vunpackb(v64);
  // CHECK: @llvm.hexagon.V6.vunpackh
  __builtin_HEXAGON_V6_vunpackh(v64);
  // CHECK: @llvm.hexagon.V6.vunpackob
  __builtin_HEXAGON_V6_vunpackob(v128, v64);
  // CHECK: @llvm.hexagon.V6.vunpackoh
  __builtin_HEXAGON_V6_vunpackoh(v128, v64);
  // CHECK: @llvm.hexagon.V6.vunpackub
  __builtin_HEXAGON_V6_vunpackub(v64);
  // CHECK: @llvm.hexagon.V6.vunpackuh
  __builtin_HEXAGON_V6_vunpackuh(v64);
  // CHECK: @llvm.hexagon.V6.vxor
  __builtin_HEXAGON_V6_vxor(v64, v64);
  // CHECK: @llvm.hexagon.V6.vzb
  __builtin_HEXAGON_V6_vzb(v64);
  // CHECK: @llvm.hexagon.V6.vzh
  __builtin_HEXAGON_V6_vzh(v64);
}
