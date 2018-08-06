// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv65 -target-feature +hvxv65 -target-feature +hvx-length128b -emit-llvm %s -o - | FileCheck %s

void test() {
  int v128 __attribute__((__vector_size__(128)));
  int v256 __attribute__((__vector_size__(256)));

  // CHECK: @llvm.hexagon.V6.extractw.128B
  __builtin_HEXAGON_V6_extractw_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.hi.128B
  __builtin_HEXAGON_V6_hi_128B(v256);
  // CHECK: @llvm.hexagon.V6.lo.128B
  __builtin_HEXAGON_V6_lo_128B(v256);
  // CHECK: @llvm.hexagon.V6.lvsplatb.128B
  __builtin_HEXAGON_V6_lvsplatb_128B(0);
  // CHECK: @llvm.hexagon.V6.lvsplath.128B
  __builtin_HEXAGON_V6_lvsplath_128B(0);
  // CHECK: @llvm.hexagon.V6.lvsplatw.128B
  __builtin_HEXAGON_V6_lvsplatw_128B(0);
  // CHECK: @llvm.hexagon.V6.pred.and.128B
  __builtin_HEXAGON_V6_pred_and_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.pred.and.n.128B
  __builtin_HEXAGON_V6_pred_and_n_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.pred.not.128B
  __builtin_HEXAGON_V6_pred_not_128B(v128);
  // CHECK: @llvm.hexagon.V6.pred.or.128B
  __builtin_HEXAGON_V6_pred_or_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.pred.or.n.128B
  __builtin_HEXAGON_V6_pred_or_n_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.pred.scalar2.128B
  __builtin_HEXAGON_V6_pred_scalar2_128B(0);
  // CHECK: @llvm.hexagon.V6.pred.scalar2v2.128B
  __builtin_HEXAGON_V6_pred_scalar2v2_128B(0);
  // CHECK: @llvm.hexagon.V6.pred.xor.128B
  __builtin_HEXAGON_V6_pred_xor_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.shuffeqh.128B
  __builtin_HEXAGON_V6_shuffeqh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.shuffeqw.128B
  __builtin_HEXAGON_V6_shuffeqw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vS32b.nqpred.ai.128B
  __builtin_HEXAGON_V6_vS32b_nqpred_ai_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vS32b.nt.nqpred.ai.128B
  __builtin_HEXAGON_V6_vS32b_nt_nqpred_ai_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vS32b.nt.qpred.ai.128B
  __builtin_HEXAGON_V6_vS32b_nt_qpred_ai_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vS32b.qpred.ai.128B
  __builtin_HEXAGON_V6_vS32b_qpred_ai_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vabsb.128B
  __builtin_HEXAGON_V6_vabsb_128B(v128);
  // CHECK: @llvm.hexagon.V6.vabsb.sat.128B
  __builtin_HEXAGON_V6_vabsb_sat_128B(v128);
  // CHECK: @llvm.hexagon.V6.vabsdiffh.128B
  __builtin_HEXAGON_V6_vabsdiffh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vabsdiffub.128B
  __builtin_HEXAGON_V6_vabsdiffub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vabsdiffuh.128B
  __builtin_HEXAGON_V6_vabsdiffuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vabsdiffw.128B
  __builtin_HEXAGON_V6_vabsdiffw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vabsh.128B
  __builtin_HEXAGON_V6_vabsh_128B(v128);
  // CHECK: @llvm.hexagon.V6.vabsh.sat.128B
  __builtin_HEXAGON_V6_vabsh_sat_128B(v128);
  // CHECK: @llvm.hexagon.V6.vabsw.128B
  __builtin_HEXAGON_V6_vabsw_128B(v128);
  // CHECK: @llvm.hexagon.V6.vabsw.sat.128B
  __builtin_HEXAGON_V6_vabsw_sat_128B(v128);
  // CHECK: @llvm.hexagon.V6.vaddb.128B
  __builtin_HEXAGON_V6_vaddb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddb.dv.128B
  __builtin_HEXAGON_V6_vaddb_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vaddbnq.128B
  __builtin_HEXAGON_V6_vaddbnq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddbq.128B
  __builtin_HEXAGON_V6_vaddbq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddbsat.128B
  __builtin_HEXAGON_V6_vaddbsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddbsat.dv.128B
  __builtin_HEXAGON_V6_vaddbsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vaddcarry.128B
  __builtin_HEXAGON_V6_vaddcarry_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vaddclbh.128B
  __builtin_HEXAGON_V6_vaddclbh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddclbw.128B
  __builtin_HEXAGON_V6_vaddclbw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddh.128B
  __builtin_HEXAGON_V6_vaddh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddh.dv.128B
  __builtin_HEXAGON_V6_vaddh_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vaddhnq.128B
  __builtin_HEXAGON_V6_vaddhnq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddhq.128B
  __builtin_HEXAGON_V6_vaddhq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddhsat.128B
  __builtin_HEXAGON_V6_vaddhsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddhsat.dv.128B
  __builtin_HEXAGON_V6_vaddhsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vaddhw.128B
  __builtin_HEXAGON_V6_vaddhw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddhw.acc.128B
  __builtin_HEXAGON_V6_vaddhw_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddubh.128B
  __builtin_HEXAGON_V6_vaddubh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddubh.acc.128B
  __builtin_HEXAGON_V6_vaddubh_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddubsat.128B
  __builtin_HEXAGON_V6_vaddubsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddubsat.dv.128B
  __builtin_HEXAGON_V6_vaddubsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vaddububb.sat.128B
  __builtin_HEXAGON_V6_vaddububb_sat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vadduhsat.128B
  __builtin_HEXAGON_V6_vadduhsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vadduhsat.dv.128B
  __builtin_HEXAGON_V6_vadduhsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vadduhw.128B
  __builtin_HEXAGON_V6_vadduhw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vadduhw.acc.128B
  __builtin_HEXAGON_V6_vadduhw_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vadduwsat.128B
  __builtin_HEXAGON_V6_vadduwsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vadduwsat.dv.128B
  __builtin_HEXAGON_V6_vadduwsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vaddw.128B
  __builtin_HEXAGON_V6_vaddw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddw.dv.128B
  __builtin_HEXAGON_V6_vaddw_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vaddwnq.128B
  __builtin_HEXAGON_V6_vaddwnq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddwq.128B
  __builtin_HEXAGON_V6_vaddwq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddwsat.128B
  __builtin_HEXAGON_V6_vaddwsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaddwsat.dv.128B
  __builtin_HEXAGON_V6_vaddwsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.valignb.128B
  __builtin_HEXAGON_V6_valignb_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.valignbi.128B
  __builtin_HEXAGON_V6_valignbi_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vand.128B
  __builtin_HEXAGON_V6_vand_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vandnqrt.128B
  __builtin_HEXAGON_V6_vandnqrt_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vandnqrt.acc.128B
  __builtin_HEXAGON_V6_vandnqrt_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt.128B
  __builtin_HEXAGON_V6_vandqrt_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt.acc.128B
  __builtin_HEXAGON_V6_vandqrt_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vandvnqv.128B
  __builtin_HEXAGON_V6_vandvnqv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vandvqv.128B
  __builtin_HEXAGON_V6_vandvqv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vandvrt.128B
  __builtin_HEXAGON_V6_vandvrt_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vandvrt.acc.128B
  __builtin_HEXAGON_V6_vandvrt_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vaslh.128B
  __builtin_HEXAGON_V6_vaslh_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vaslh.acc.128B
  __builtin_HEXAGON_V6_vaslh_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vaslhv.128B
  __builtin_HEXAGON_V6_vaslhv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vaslw.128B
  __builtin_HEXAGON_V6_vaslw_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vaslw.acc.128B
  __builtin_HEXAGON_V6_vaslw_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vaslwv.128B
  __builtin_HEXAGON_V6_vaslwv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vasrh.128B
  __builtin_HEXAGON_V6_vasrh_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrh.acc.128B
  __builtin_HEXAGON_V6_vasrh_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrhbrndsat.128B
  __builtin_HEXAGON_V6_vasrhbrndsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrhbsat.128B
  __builtin_HEXAGON_V6_vasrhbsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubrndsat.128B
  __builtin_HEXAGON_V6_vasrhubrndsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubsat.128B
  __builtin_HEXAGON_V6_vasrhubsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrhv.128B
  __builtin_HEXAGON_V6_vasrhv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vasruhubrndsat.128B
  __builtin_HEXAGON_V6_vasruhubrndsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasruhubsat.128B
  __builtin_HEXAGON_V6_vasruhubsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasruwuhrndsat.128B
  __builtin_HEXAGON_V6_vasruwuhrndsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasruwuhsat.128B
  __builtin_HEXAGON_V6_vasruwuhsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrw.128B
  __builtin_HEXAGON_V6_vasrw_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrw.acc.128B
  __builtin_HEXAGON_V6_vasrw_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrwh.128B
  __builtin_HEXAGON_V6_vasrwh_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhrndsat.128B
  __builtin_HEXAGON_V6_vasrwhrndsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhsat.128B
  __builtin_HEXAGON_V6_vasrwhsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrwuhrndsat.128B
  __builtin_HEXAGON_V6_vasrwuhrndsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrwuhsat.128B
  __builtin_HEXAGON_V6_vasrwuhsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vasrwv.128B
  __builtin_HEXAGON_V6_vasrwv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vassign.128B
  __builtin_HEXAGON_V6_vassign_128B(v128);
  // CHECK: @llvm.hexagon.V6.vassignp.128B
  __builtin_HEXAGON_V6_vassignp_128B(v256);
  // CHECK: @llvm.hexagon.V6.vavgb.128B
  __builtin_HEXAGON_V6_vavgb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavgbrnd.128B
  __builtin_HEXAGON_V6_vavgbrnd_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavgh.128B
  __builtin_HEXAGON_V6_vavgh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavghrnd.128B
  __builtin_HEXAGON_V6_vavghrnd_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavgub.128B
  __builtin_HEXAGON_V6_vavgub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavgubrnd.128B
  __builtin_HEXAGON_V6_vavgubrnd_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavguh.128B
  __builtin_HEXAGON_V6_vavguh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavguhrnd.128B
  __builtin_HEXAGON_V6_vavguhrnd_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavguw.128B
  __builtin_HEXAGON_V6_vavguw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavguwrnd.128B
  __builtin_HEXAGON_V6_vavguwrnd_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavgw.128B
  __builtin_HEXAGON_V6_vavgw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vavgwrnd.128B
  __builtin_HEXAGON_V6_vavgwrnd_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vcl0h.128B
  __builtin_HEXAGON_V6_vcl0h_128B(v128);
  // CHECK: @llvm.hexagon.V6.vcl0w.128B
  __builtin_HEXAGON_V6_vcl0w_128B(v128);
  // CHECK: @llvm.hexagon.V6.vcombine.128B
  __builtin_HEXAGON_V6_vcombine_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vd0.128B
  __builtin_HEXAGON_V6_vd0_128B();
  // CHECK: @llvm.hexagon.V6.vdd0.128B
  __builtin_HEXAGON_V6_vdd0_128B();
  // CHECK: @llvm.hexagon.V6.vdealb.128B
  __builtin_HEXAGON_V6_vdealb_128B(v128);
  // CHECK: @llvm.hexagon.V6.vdealb4w.128B
  __builtin_HEXAGON_V6_vdealb4w_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vdealh.128B
  __builtin_HEXAGON_V6_vdealh_128B(v128);
  // CHECK: @llvm.hexagon.V6.vdealvdd.128B
  __builtin_HEXAGON_V6_vdealvdd_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdelta.128B
  __builtin_HEXAGON_V6_vdelta_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vdmpybus.128B
  __builtin_HEXAGON_V6_vdmpybus_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.acc.128B
  __builtin_HEXAGON_V6_vdmpybus_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv.128B
  __builtin_HEXAGON_V6_vdmpybus_dv_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv.acc.128B
  __builtin_HEXAGON_V6_vdmpybus_dv_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.128B
  __builtin_HEXAGON_V6_vdmpyhb_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.acc.128B
  __builtin_HEXAGON_V6_vdmpyhb_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv.128B
  __builtin_HEXAGON_V6_vdmpyhb_dv_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv.acc.128B
  __builtin_HEXAGON_V6_vdmpyhb_dv_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat.128B
  __builtin_HEXAGON_V6_vdmpyhisat_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhisat_acc_128B(v128, v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat.128B
  __builtin_HEXAGON_V6_vdmpyhsat_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhsat_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat.128B
  __builtin_HEXAGON_V6_vdmpyhsuisat_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhsuisat_acc_128B(v128, v256, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat.128B
  __builtin_HEXAGON_V6_vdmpyhsusat_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhsusat_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat.128B
  __builtin_HEXAGON_V6_vdmpyhvsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vdsaduh.128B
  __builtin_HEXAGON_V6_vdsaduh_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vdsaduh.acc.128B
  __builtin_HEXAGON_V6_vdsaduh_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.veqb.128B
  __builtin_HEXAGON_V6_veqb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.veqb.and.128B
  __builtin_HEXAGON_V6_veqb_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqb.or.128B
  __builtin_HEXAGON_V6_veqb_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqb.xor.128B
  __builtin_HEXAGON_V6_veqb_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqh.128B
  __builtin_HEXAGON_V6_veqh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.veqh.and.128B
  __builtin_HEXAGON_V6_veqh_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqh.or.128B
  __builtin_HEXAGON_V6_veqh_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqh.xor.128B
  __builtin_HEXAGON_V6_veqh_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqw.128B
  __builtin_HEXAGON_V6_veqw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.veqw.and.128B
  __builtin_HEXAGON_V6_veqw_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqw.or.128B
  __builtin_HEXAGON_V6_veqw_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.veqw.xor.128B
  __builtin_HEXAGON_V6_veqw_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgathermh.128B
  __builtin_HEXAGON_V6_vgathermh_128B(0, 0, 0, v128);
  // CHECK: @llvm.hexagon.V6.vgathermhq.128B
  __builtin_HEXAGON_V6_vgathermhq_128B(0, v128, 0, 0, v128);
  // CHECK: @llvm.hexagon.V6.vgathermhw.128B
  __builtin_HEXAGON_V6_vgathermhw_128B(0, 0, 0, v256);
  // CHECK: @llvm.hexagon.V6.vgathermhwq.128B
  __builtin_HEXAGON_V6_vgathermhwq_128B(0, v128, 0, 0, v256);
  // CHECK: @llvm.hexagon.V6.vgathermw.128B
  __builtin_HEXAGON_V6_vgathermw_128B(0, 0, 0, v128);
  // CHECK: @llvm.hexagon.V6.vgathermwq.128B
  __builtin_HEXAGON_V6_vgathermwq_128B(0, v128, 0, 0, v128);
  // CHECK: @llvm.hexagon.V6.vgtb.128B
  __builtin_HEXAGON_V6_vgtb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtb.and.128B
  __builtin_HEXAGON_V6_vgtb_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtb.or.128B
  __builtin_HEXAGON_V6_vgtb_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtb.xor.128B
  __builtin_HEXAGON_V6_vgtb_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgth.128B
  __builtin_HEXAGON_V6_vgth_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vgth.and.128B
  __builtin_HEXAGON_V6_vgth_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgth.or.128B
  __builtin_HEXAGON_V6_vgth_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgth.xor.128B
  __builtin_HEXAGON_V6_vgth_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtub.128B
  __builtin_HEXAGON_V6_vgtub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtub.and.128B
  __builtin_HEXAGON_V6_vgtub_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtub.or.128B
  __builtin_HEXAGON_V6_vgtub_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtub.xor.128B
  __builtin_HEXAGON_V6_vgtub_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuh.128B
  __builtin_HEXAGON_V6_vgtuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuh.and.128B
  __builtin_HEXAGON_V6_vgtuh_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuh.or.128B
  __builtin_HEXAGON_V6_vgtuh_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuh.xor.128B
  __builtin_HEXAGON_V6_vgtuh_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuw.128B
  __builtin_HEXAGON_V6_vgtuw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuw.and.128B
  __builtin_HEXAGON_V6_vgtuw_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuw.or.128B
  __builtin_HEXAGON_V6_vgtuw_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtuw.xor.128B
  __builtin_HEXAGON_V6_vgtuw_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtw.128B
  __builtin_HEXAGON_V6_vgtw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtw.and.128B
  __builtin_HEXAGON_V6_vgtw_and_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtw.or.128B
  __builtin_HEXAGON_V6_vgtw_or_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vgtw.xor.128B
  __builtin_HEXAGON_V6_vgtw_xor_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vinsertwr.128B
  __builtin_HEXAGON_V6_vinsertwr_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vlalignb.128B
  __builtin_HEXAGON_V6_vlalignb_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlalignbi.128B
  __builtin_HEXAGON_V6_vlalignbi_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlsrb.128B
  __builtin_HEXAGON_V6_vlsrb_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vlsrh.128B
  __builtin_HEXAGON_V6_vlsrh_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vlsrhv.128B
  __builtin_HEXAGON_V6_vlsrhv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vlsrw.128B
  __builtin_HEXAGON_V6_vlsrw_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vlsrwv.128B
  __builtin_HEXAGON_V6_vlsrwv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vlut4.128B
  __builtin_HEXAGON_V6_vlut4_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.128B
  __builtin_HEXAGON_V6_vlutvvb_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.nm.128B
  __builtin_HEXAGON_V6_vlutvvb_nm_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.oracc.128B
  __builtin_HEXAGON_V6_vlutvvb_oracc_128B(v128, v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.oracci.128B
  __builtin_HEXAGON_V6_vlutvvb_oracci_128B(v128, v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvbi.128B
  __builtin_HEXAGON_V6_vlutvvbi_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.128B
  __builtin_HEXAGON_V6_vlutvwh_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.nm.128B
  __builtin_HEXAGON_V6_vlutvwh_nm_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.oracc.128B
  __builtin_HEXAGON_V6_vlutvwh_oracc_128B(v256, v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.oracci.128B
  __builtin_HEXAGON_V6_vlutvwh_oracci_128B(v256, v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwhi.128B
  __builtin_HEXAGON_V6_vlutvwhi_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmaskedstorenq.128B
  __builtin_HEXAGON_V6_vmaskedstorenq_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vmaskedstorentnq.128B
  __builtin_HEXAGON_V6_vmaskedstorentnq_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vmaskedstorentq.128B
  __builtin_HEXAGON_V6_vmaskedstorentq_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vmaskedstoreq.128B
  __builtin_HEXAGON_V6_vmaskedstoreq_128B(v128, 0, v128);
  // CHECK: @llvm.hexagon.V6.vmaxb.128B
  __builtin_HEXAGON_V6_vmaxb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmaxh.128B
  __builtin_HEXAGON_V6_vmaxh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmaxub.128B
  __builtin_HEXAGON_V6_vmaxub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmaxuh.128B
  __builtin_HEXAGON_V6_vmaxuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmaxw.128B
  __builtin_HEXAGON_V6_vmaxw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vminb.128B
  __builtin_HEXAGON_V6_vminb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vminh.128B
  __builtin_HEXAGON_V6_vminh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vminub.128B
  __builtin_HEXAGON_V6_vminub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vminuh.128B
  __builtin_HEXAGON_V6_vminuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vminw.128B
  __builtin_HEXAGON_V6_vminw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpabus.128B
  __builtin_HEXAGON_V6_vmpabus_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpabus.acc.128B
  __builtin_HEXAGON_V6_vmpabus_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpabusv.128B
  __builtin_HEXAGON_V6_vmpabusv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vmpabuu.128B
  __builtin_HEXAGON_V6_vmpabuu_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpabuu.acc.128B
  __builtin_HEXAGON_V6_vmpabuu_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpabuuv.128B
  __builtin_HEXAGON_V6_vmpabuuv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vmpahb.128B
  __builtin_HEXAGON_V6_vmpahb_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpahb.acc.128B
  __builtin_HEXAGON_V6_vmpahb_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpahhsat.128B
  __builtin_HEXAGON_V6_vmpahhsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpauhb.128B
  __builtin_HEXAGON_V6_vmpauhb_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpauhb.acc.128B
  __builtin_HEXAGON_V6_vmpauhb_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vmpauhuhsat.128B
  __builtin_HEXAGON_V6_vmpauhuhsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpsuhuhsat.128B
  __builtin_HEXAGON_V6_vmpsuhuhsat_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus.128B
  __builtin_HEXAGON_V6_vmpybus_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus.acc.128B
  __builtin_HEXAGON_V6_vmpybus_acc_128B(v256, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpybusv.128B
  __builtin_HEXAGON_V6_vmpybusv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpybusv.acc.128B
  __builtin_HEXAGON_V6_vmpybusv_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpybv.128B
  __builtin_HEXAGON_V6_vmpybv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpybv.acc.128B
  __builtin_HEXAGON_V6_vmpybv_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyewuh.128B
  __builtin_HEXAGON_V6_vmpyewuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyewuh.64.128B
  __builtin_HEXAGON_V6_vmpyewuh_64_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyh.128B
  __builtin_HEXAGON_V6_vmpyh_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyh.acc.128B
  __builtin_HEXAGON_V6_vmpyh_acc_128B(v256, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsat.acc.128B
  __builtin_HEXAGON_V6_vmpyhsat_acc_128B(v256, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsrs.128B
  __builtin_HEXAGON_V6_vmpyhsrs_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhss.128B
  __builtin_HEXAGON_V6_vmpyhss_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhus.128B
  __builtin_HEXAGON_V6_vmpyhus_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyhus.acc.128B
  __builtin_HEXAGON_V6_vmpyhus_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyhv.128B
  __builtin_HEXAGON_V6_vmpyhv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyhv.acc.128B
  __builtin_HEXAGON_V6_vmpyhv_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyhvsrs.128B
  __builtin_HEXAGON_V6_vmpyhvsrs_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyieoh.128B
  __builtin_HEXAGON_V6_vmpyieoh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyiewh.acc.128B
  __builtin_HEXAGON_V6_vmpyiewh_acc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh.128B
  __builtin_HEXAGON_V6_vmpyiewuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh.acc.128B
  __builtin_HEXAGON_V6_vmpyiewuh_acc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyih.128B
  __builtin_HEXAGON_V6_vmpyih_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyih.acc.128B
  __builtin_HEXAGON_V6_vmpyih_acc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyihb.128B
  __builtin_HEXAGON_V6_vmpyihb_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyihb.acc.128B
  __builtin_HEXAGON_V6_vmpyihb_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiowh.128B
  __builtin_HEXAGON_V6_vmpyiowh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyiwb.128B
  __builtin_HEXAGON_V6_vmpyiwb_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwb.acc.128B
  __builtin_HEXAGON_V6_vmpyiwb_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh.128B
  __builtin_HEXAGON_V6_vmpyiwh_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh.acc.128B
  __builtin_HEXAGON_V6_vmpyiwh_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwub.128B
  __builtin_HEXAGON_V6_vmpyiwub_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwub.acc.128B
  __builtin_HEXAGON_V6_vmpyiwub_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyowh.128B
  __builtin_HEXAGON_V6_vmpyowh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyowh.64.acc.128B
  __builtin_HEXAGON_V6_vmpyowh_64_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd.128B
  __builtin_HEXAGON_V6_vmpyowh_rnd_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd.sacc.128B
  __builtin_HEXAGON_V6_vmpyowh_rnd_sacc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyowh.sacc.128B
  __builtin_HEXAGON_V6_vmpyowh_sacc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyub.128B
  __builtin_HEXAGON_V6_vmpyub_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyub.acc.128B
  __builtin_HEXAGON_V6_vmpyub_acc_128B(v256, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyubv.128B
  __builtin_HEXAGON_V6_vmpyubv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyubv.acc.128B
  __builtin_HEXAGON_V6_vmpyubv_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyuh.128B
  __builtin_HEXAGON_V6_vmpyuh_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuh.acc.128B
  __builtin_HEXAGON_V6_vmpyuh_acc_128B(v256, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuhe.128B
  __builtin_HEXAGON_V6_vmpyuhe_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuhe.acc.128B
  __builtin_HEXAGON_V6_vmpyuhe_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuhv.128B
  __builtin_HEXAGON_V6_vmpyuhv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vmpyuhv.acc.128B
  __builtin_HEXAGON_V6_vmpyuhv_acc_128B(v256, v128, v128);
  // CHECK: @llvm.hexagon.V6.vmux.128B
  __builtin_HEXAGON_V6_vmux_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vnavgb.128B
  __builtin_HEXAGON_V6_vnavgb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vnavgh.128B
  __builtin_HEXAGON_V6_vnavgh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vnavgub.128B
  __builtin_HEXAGON_V6_vnavgub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vnavgw.128B
  __builtin_HEXAGON_V6_vnavgw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vnormamth.128B
  __builtin_HEXAGON_V6_vnormamth_128B(v128);
  // CHECK: @llvm.hexagon.V6.vnormamtw.128B
  __builtin_HEXAGON_V6_vnormamtw_128B(v128);
  // CHECK: @llvm.hexagon.V6.vnot.128B
  __builtin_HEXAGON_V6_vnot_128B(v128);
  // CHECK: @llvm.hexagon.V6.vor.128B
  __builtin_HEXAGON_V6_vor_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackeb.128B
  __builtin_HEXAGON_V6_vpackeb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackeh.128B
  __builtin_HEXAGON_V6_vpackeh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackhb.sat.128B
  __builtin_HEXAGON_V6_vpackhb_sat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackhub.sat.128B
  __builtin_HEXAGON_V6_vpackhub_sat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackob.128B
  __builtin_HEXAGON_V6_vpackob_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackoh.128B
  __builtin_HEXAGON_V6_vpackoh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackwh.sat.128B
  __builtin_HEXAGON_V6_vpackwh_sat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpackwuh.sat.128B
  __builtin_HEXAGON_V6_vpackwuh_sat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vpopcounth.128B
  __builtin_HEXAGON_V6_vpopcounth_128B(v128);
  // CHECK: @llvm.hexagon.V6.vprefixqb.128B
  __builtin_HEXAGON_V6_vprefixqb_128B(v128);
  // CHECK: @llvm.hexagon.V6.vprefixqh.128B
  __builtin_HEXAGON_V6_vprefixqh_128B(v128);
  // CHECK: @llvm.hexagon.V6.vprefixqw.128B
  __builtin_HEXAGON_V6_vprefixqw_128B(v128);
  // CHECK: @llvm.hexagon.V6.vrdelta.128B
  __builtin_HEXAGON_V6_vrdelta_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vrmpybub.rtt.128B
  __builtin_HEXAGON_V6_vrmpybub_rtt_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybub.rtt.acc.128B
  __builtin_HEXAGON_V6_vrmpybub_rtt_acc_128B(v256, v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus.128B
  __builtin_HEXAGON_V6_vrmpybus_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus.acc.128B
  __builtin_HEXAGON_V6_vrmpybus_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi.128B
  __builtin_HEXAGON_V6_vrmpybusi_128B(v256, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi.acc.128B
  __builtin_HEXAGON_V6_vrmpybusi_acc_128B(v256, v256, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusv.128B
  __builtin_HEXAGON_V6_vrmpybusv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vrmpybusv.acc.128B
  __builtin_HEXAGON_V6_vrmpybusv_acc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vrmpybv.128B
  __builtin_HEXAGON_V6_vrmpybv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vrmpybv.acc.128B
  __builtin_HEXAGON_V6_vrmpybv_acc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vrmpyub.128B
  __builtin_HEXAGON_V6_vrmpyub_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.acc.128B
  __builtin_HEXAGON_V6_vrmpyub_acc_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.rtt.128B
  __builtin_HEXAGON_V6_vrmpyub_rtt_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.rtt.acc.128B
  __builtin_HEXAGON_V6_vrmpyub_rtt_acc_128B(v256, v128, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi.128B
  __builtin_HEXAGON_V6_vrmpyubi_128B(v256, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi.acc.128B
  __builtin_HEXAGON_V6_vrmpyubi_acc_128B(v256, v256, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubv.128B
  __builtin_HEXAGON_V6_vrmpyubv_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vrmpyubv.acc.128B
  __builtin_HEXAGON_V6_vrmpyubv_acc_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vror.128B
  __builtin_HEXAGON_V6_vror_128B(v128, 0);
  // CHECK: @llvm.hexagon.V6.vroundhb.128B
  __builtin_HEXAGON_V6_vroundhb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vroundhub.128B
  __builtin_HEXAGON_V6_vroundhub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vrounduhub.128B
  __builtin_HEXAGON_V6_vrounduhub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vrounduwuh.128B
  __builtin_HEXAGON_V6_vrounduwuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vroundwh.128B
  __builtin_HEXAGON_V6_vroundwh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vroundwuh.128B
  __builtin_HEXAGON_V6_vroundwuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vrsadubi.128B
  __builtin_HEXAGON_V6_vrsadubi_128B(v256, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrsadubi.acc.128B
  __builtin_HEXAGON_V6_vrsadubi_acc_128B(v256, v256, 0, 0);
  // CHECK: @llvm.hexagon.V6.vsathub.128B
  __builtin_HEXAGON_V6_vsathub_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsatuwuh.128B
  __builtin_HEXAGON_V6_vsatuwuh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsatwh.128B
  __builtin_HEXAGON_V6_vsatwh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsb.128B
  __builtin_HEXAGON_V6_vsb_128B(v128);
  // CHECK: @llvm.hexagon.V6.vscattermh.128B
  __builtin_HEXAGON_V6_vscattermh_128B(0, 0, v128, v128);
  // CHECK: @llvm.hexagon.V6.vscattermh.add.128B
  __builtin_HEXAGON_V6_vscattermh_add_128B(0, 0, v128, v128);
  // CHECK: @llvm.hexagon.V6.vscattermhq.128B
  __builtin_HEXAGON_V6_vscattermhq_128B(v128, 0, 0, v128, v128);
  // CHECK: @llvm.hexagon.V6.vscattermhw.128B
  __builtin_HEXAGON_V6_vscattermhw_128B(0, 0, v256, v128);
  // CHECK: @llvm.hexagon.V6.vscattermhw.add.128B
  __builtin_HEXAGON_V6_vscattermhw_add_128B(0, 0, v256, v128);
  // CHECK: @llvm.hexagon.V6.vscattermhwq.128B
  __builtin_HEXAGON_V6_vscattermhwq_128B(v128, 0, 0, v256, v128);
  // CHECK: @llvm.hexagon.V6.vscattermw.128B
  __builtin_HEXAGON_V6_vscattermw_128B(0, 0, v128, v128);
  // CHECK: @llvm.hexagon.V6.vscattermw.add.128B
  __builtin_HEXAGON_V6_vscattermw_add_128B(0, 0, v128, v128);
  // CHECK: @llvm.hexagon.V6.vscattermwq.128B
  __builtin_HEXAGON_V6_vscattermwq_128B(v128, 0, 0, v128, v128);
  // CHECK: @llvm.hexagon.V6.vsh.128B
  __builtin_HEXAGON_V6_vsh_128B(v128);
  // CHECK: @llvm.hexagon.V6.vshufeh.128B
  __builtin_HEXAGON_V6_vshufeh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vshuffb.128B
  __builtin_HEXAGON_V6_vshuffb_128B(v128);
  // CHECK: @llvm.hexagon.V6.vshuffeb.128B
  __builtin_HEXAGON_V6_vshuffeb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vshuffh.128B
  __builtin_HEXAGON_V6_vshuffh_128B(v128);
  // CHECK: @llvm.hexagon.V6.vshuffob.128B
  __builtin_HEXAGON_V6_vshuffob_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vshuffvdd.128B
  __builtin_HEXAGON_V6_vshuffvdd_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vshufoeb.128B
  __builtin_HEXAGON_V6_vshufoeb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vshufoeh.128B
  __builtin_HEXAGON_V6_vshufoeh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vshufoh.128B
  __builtin_HEXAGON_V6_vshufoh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubb.128B
  __builtin_HEXAGON_V6_vsubb_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubb.dv.128B
  __builtin_HEXAGON_V6_vsubb_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubbnq.128B
  __builtin_HEXAGON_V6_vsubbnq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubbq.128B
  __builtin_HEXAGON_V6_vsubbq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubbsat.128B
  __builtin_HEXAGON_V6_vsubbsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubbsat.dv.128B
  __builtin_HEXAGON_V6_vsubbsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubcarry.128B
  __builtin_HEXAGON_V6_vsubcarry_128B(v128, v128, 0);
  // CHECK: @llvm.hexagon.V6.vsubh.128B
  __builtin_HEXAGON_V6_vsubh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubh.dv.128B
  __builtin_HEXAGON_V6_vsubh_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubhnq.128B
  __builtin_HEXAGON_V6_vsubhnq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubhq.128B
  __builtin_HEXAGON_V6_vsubhq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubhsat.128B
  __builtin_HEXAGON_V6_vsubhsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubhsat.dv.128B
  __builtin_HEXAGON_V6_vsubhsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubhw.128B
  __builtin_HEXAGON_V6_vsubhw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsububh.128B
  __builtin_HEXAGON_V6_vsububh_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsububsat.128B
  __builtin_HEXAGON_V6_vsububsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsububsat.dv.128B
  __builtin_HEXAGON_V6_vsububsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubububb.sat.128B
  __builtin_HEXAGON_V6_vsubububb_sat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubuhsat.128B
  __builtin_HEXAGON_V6_vsubuhsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubuhsat.dv.128B
  __builtin_HEXAGON_V6_vsubuhsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubuhw.128B
  __builtin_HEXAGON_V6_vsubuhw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubuwsat.128B
  __builtin_HEXAGON_V6_vsubuwsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubuwsat.dv.128B
  __builtin_HEXAGON_V6_vsubuwsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubw.128B
  __builtin_HEXAGON_V6_vsubw_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubw.dv.128B
  __builtin_HEXAGON_V6_vsubw_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vsubwnq.128B
  __builtin_HEXAGON_V6_vsubwnq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubwq.128B
  __builtin_HEXAGON_V6_vsubwq_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubwsat.128B
  __builtin_HEXAGON_V6_vsubwsat_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vsubwsat.dv.128B
  __builtin_HEXAGON_V6_vsubwsat_dv_128B(v256, v256);
  // CHECK: @llvm.hexagon.V6.vswap.128B
  __builtin_HEXAGON_V6_vswap_128B(v128, v128, v128);
  // CHECK: @llvm.hexagon.V6.vtmpyb.128B
  __builtin_HEXAGON_V6_vtmpyb_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyb.acc.128B
  __builtin_HEXAGON_V6_vtmpyb_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus.128B
  __builtin_HEXAGON_V6_vtmpybus_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus.acc.128B
  __builtin_HEXAGON_V6_vtmpybus_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb.128B
  __builtin_HEXAGON_V6_vtmpyhb_128B(v256, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb.acc.128B
  __builtin_HEXAGON_V6_vtmpyhb_acc_128B(v256, v256, 0);
  // CHECK: @llvm.hexagon.V6.vunpackb.128B
  __builtin_HEXAGON_V6_vunpackb_128B(v128);
  // CHECK: @llvm.hexagon.V6.vunpackh.128B
  __builtin_HEXAGON_V6_vunpackh_128B(v128);
  // CHECK: @llvm.hexagon.V6.vunpackob.128B
  __builtin_HEXAGON_V6_vunpackob_128B(v256, v128);
  // CHECK: @llvm.hexagon.V6.vunpackoh.128B
  __builtin_HEXAGON_V6_vunpackoh_128B(v256, v128);
  // CHECK: @llvm.hexagon.V6.vunpackub.128B
  __builtin_HEXAGON_V6_vunpackub_128B(v128);
  // CHECK: @llvm.hexagon.V6.vunpackuh.128B
  __builtin_HEXAGON_V6_vunpackuh_128B(v128);
  // CHECK: @llvm.hexagon.V6.vxor.128B
  __builtin_HEXAGON_V6_vxor_128B(v128, v128);
  // CHECK: @llvm.hexagon.V6.vzb.128B
  __builtin_HEXAGON_V6_vzb_128B(v128);
  // CHECK: @llvm.hexagon.V6.vzh.128B
  __builtin_HEXAGON_V6_vzh_128B(v128);
}
