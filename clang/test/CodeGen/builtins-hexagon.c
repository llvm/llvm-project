// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv65 -target-feature +hvxv65 -emit-llvm %s -o - | FileCheck %s

void test() {
  int v64 __attribute__((__vector_size__(64)));
  int v128 __attribute__((__vector_size__(128)));
  int v256 __attribute__((__vector_size__(256)));

  // CHECK: @llvm.hexagon.A2.abs
  __builtin_HEXAGON_A2_abs(0);
  // CHECK: @llvm.hexagon.A2.absp
  __builtin_HEXAGON_A2_absp(0);
  // CHECK: @llvm.hexagon.A2.abssat
  __builtin_HEXAGON_A2_abssat(0);
  // CHECK: @llvm.hexagon.A2.add
  __builtin_HEXAGON_A2_add(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.hh
  __builtin_HEXAGON_A2_addh_h16_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.hl
  __builtin_HEXAGON_A2_addh_h16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.lh
  __builtin_HEXAGON_A2_addh_h16_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.ll
  __builtin_HEXAGON_A2_addh_h16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.hh
  __builtin_HEXAGON_A2_addh_h16_sat_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.hl
  __builtin_HEXAGON_A2_addh_h16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.lh
  __builtin_HEXAGON_A2_addh_h16_sat_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.ll
  __builtin_HEXAGON_A2_addh_h16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.hl
  __builtin_HEXAGON_A2_addh_l16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.ll
  __builtin_HEXAGON_A2_addh_l16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.sat.hl
  __builtin_HEXAGON_A2_addh_l16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.sat.ll
  __builtin_HEXAGON_A2_addh_l16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addi
  __builtin_HEXAGON_A2_addi(0, 0);
  // CHECK: @llvm.hexagon.A2.addp
  __builtin_HEXAGON_A2_addp(0, 0);
  // CHECK: @llvm.hexagon.A2.addpsat
  __builtin_HEXAGON_A2_addpsat(0, 0);
  // CHECK: @llvm.hexagon.A2.addsat
  __builtin_HEXAGON_A2_addsat(0, 0);
  // CHECK: @llvm.hexagon.A2.addsp
  __builtin_HEXAGON_A2_addsp(0, 0);
  // CHECK: @llvm.hexagon.A2.and
  __builtin_HEXAGON_A2_and(0, 0);
  // CHECK: @llvm.hexagon.A2.andir
  __builtin_HEXAGON_A2_andir(0, 0);
  // CHECK: @llvm.hexagon.A2.andp
  __builtin_HEXAGON_A2_andp(0, 0);
  // CHECK: @llvm.hexagon.A2.aslh
  __builtin_HEXAGON_A2_aslh(0);
  // CHECK: @llvm.hexagon.A2.asrh
  __builtin_HEXAGON_A2_asrh(0);
  // CHECK: @llvm.hexagon.A2.combine.hh
  __builtin_HEXAGON_A2_combine_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.combine.hl
  __builtin_HEXAGON_A2_combine_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.combine.lh
  __builtin_HEXAGON_A2_combine_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.combine.ll
  __builtin_HEXAGON_A2_combine_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.combineii
  __builtin_HEXAGON_A2_combineii(0, 0);
  // CHECK: @llvm.hexagon.A2.combinew
  __builtin_HEXAGON_A2_combinew(0, 0);
  // CHECK: @llvm.hexagon.A2.max
  __builtin_HEXAGON_A2_max(0, 0);
  // CHECK: @llvm.hexagon.A2.maxp
  __builtin_HEXAGON_A2_maxp(0, 0);
  // CHECK: @llvm.hexagon.A2.maxu
  __builtin_HEXAGON_A2_maxu(0, 0);
  // CHECK: @llvm.hexagon.A2.maxup
  __builtin_HEXAGON_A2_maxup(0, 0);
  // CHECK: @llvm.hexagon.A2.min
  __builtin_HEXAGON_A2_min(0, 0);
  // CHECK: @llvm.hexagon.A2.minp
  __builtin_HEXAGON_A2_minp(0, 0);
  // CHECK: @llvm.hexagon.A2.minu
  __builtin_HEXAGON_A2_minu(0, 0);
  // CHECK: @llvm.hexagon.A2.minup
  __builtin_HEXAGON_A2_minup(0, 0);
  // CHECK: @llvm.hexagon.A2.neg
  __builtin_HEXAGON_A2_neg(0);
  // CHECK: @llvm.hexagon.A2.negp
  __builtin_HEXAGON_A2_negp(0);
  // CHECK: @llvm.hexagon.A2.negsat
  __builtin_HEXAGON_A2_negsat(0);
  // CHECK: @llvm.hexagon.A2.not
  __builtin_HEXAGON_A2_not(0);
  // CHECK: @llvm.hexagon.A2.notp
  __builtin_HEXAGON_A2_notp(0);
  // CHECK: @llvm.hexagon.A2.or
  __builtin_HEXAGON_A2_or(0, 0);
  // CHECK: @llvm.hexagon.A2.orir
  __builtin_HEXAGON_A2_orir(0, 0);
  // CHECK: @llvm.hexagon.A2.orp
  __builtin_HEXAGON_A2_orp(0, 0);
  // CHECK: @llvm.hexagon.A2.roundsat
  __builtin_HEXAGON_A2_roundsat(0);
  // CHECK: @llvm.hexagon.A2.sat
  __builtin_HEXAGON_A2_sat(0);
  // CHECK: @llvm.hexagon.A2.satb
  __builtin_HEXAGON_A2_satb(0);
  // CHECK: @llvm.hexagon.A2.sath
  __builtin_HEXAGON_A2_sath(0);
  // CHECK: @llvm.hexagon.A2.satub
  __builtin_HEXAGON_A2_satub(0);
  // CHECK: @llvm.hexagon.A2.satuh
  __builtin_HEXAGON_A2_satuh(0);
  // CHECK: @llvm.hexagon.A2.sub
  __builtin_HEXAGON_A2_sub(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.hh
  __builtin_HEXAGON_A2_subh_h16_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.hl
  __builtin_HEXAGON_A2_subh_h16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.lh
  __builtin_HEXAGON_A2_subh_h16_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.ll
  __builtin_HEXAGON_A2_subh_h16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.hh
  __builtin_HEXAGON_A2_subh_h16_sat_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.hl
  __builtin_HEXAGON_A2_subh_h16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.lh
  __builtin_HEXAGON_A2_subh_h16_sat_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.ll
  __builtin_HEXAGON_A2_subh_h16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.hl
  __builtin_HEXAGON_A2_subh_l16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.ll
  __builtin_HEXAGON_A2_subh_l16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.sat.hl
  __builtin_HEXAGON_A2_subh_l16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.sat.ll
  __builtin_HEXAGON_A2_subh_l16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subp
  __builtin_HEXAGON_A2_subp(0, 0);
  // CHECK: @llvm.hexagon.A2.subri
  __builtin_HEXAGON_A2_subri(0, 0);
  // CHECK: @llvm.hexagon.A2.subsat
  __builtin_HEXAGON_A2_subsat(0, 0);
  // CHECK: @llvm.hexagon.A2.svaddh
  __builtin_HEXAGON_A2_svaddh(0, 0);
  // CHECK: @llvm.hexagon.A2.svaddhs
  __builtin_HEXAGON_A2_svaddhs(0, 0);
  // CHECK: @llvm.hexagon.A2.svadduhs
  __builtin_HEXAGON_A2_svadduhs(0, 0);
  // CHECK: @llvm.hexagon.A2.svavgh
  __builtin_HEXAGON_A2_svavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.svavghs
  __builtin_HEXAGON_A2_svavghs(0, 0);
  // CHECK: @llvm.hexagon.A2.svnavgh
  __builtin_HEXAGON_A2_svnavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.svsubh
  __builtin_HEXAGON_A2_svsubh(0, 0);
  // CHECK: @llvm.hexagon.A2.svsubhs
  __builtin_HEXAGON_A2_svsubhs(0, 0);
  // CHECK: @llvm.hexagon.A2.svsubuhs
  __builtin_HEXAGON_A2_svsubuhs(0, 0);
  // CHECK: @llvm.hexagon.A2.swiz
  __builtin_HEXAGON_A2_swiz(0);
  // CHECK: @llvm.hexagon.A2.sxtb
  __builtin_HEXAGON_A2_sxtb(0);
  // CHECK: @llvm.hexagon.A2.sxth
  __builtin_HEXAGON_A2_sxth(0);
  // CHECK: @llvm.hexagon.A2.sxtw
  __builtin_HEXAGON_A2_sxtw(0);
  // CHECK: @llvm.hexagon.A2.tfr
  __builtin_HEXAGON_A2_tfr(0);
  // CHECK: @llvm.hexagon.A2.tfrih
  __builtin_HEXAGON_A2_tfrih(0, 0);
  // CHECK: @llvm.hexagon.A2.tfril
  __builtin_HEXAGON_A2_tfril(0, 0);
  // CHECK: @llvm.hexagon.A2.tfrp
  __builtin_HEXAGON_A2_tfrp(0);
  // CHECK: @llvm.hexagon.A2.tfrpi
  __builtin_HEXAGON_A2_tfrpi(0);
  // CHECK: @llvm.hexagon.A2.tfrsi
  __builtin_HEXAGON_A2_tfrsi(0);
  // CHECK: @llvm.hexagon.A2.vabsh
  __builtin_HEXAGON_A2_vabsh(0);
  // CHECK: @llvm.hexagon.A2.vabshsat
  __builtin_HEXAGON_A2_vabshsat(0);
  // CHECK: @llvm.hexagon.A2.vabsw
  __builtin_HEXAGON_A2_vabsw(0);
  // CHECK: @llvm.hexagon.A2.vabswsat
  __builtin_HEXAGON_A2_vabswsat(0);
  // CHECK: @llvm.hexagon.A2.vaddb.map
  __builtin_HEXAGON_A2_vaddb_map(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddh
  __builtin_HEXAGON_A2_vaddh(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddhs
  __builtin_HEXAGON_A2_vaddhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddub
  __builtin_HEXAGON_A2_vaddub(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddubs
  __builtin_HEXAGON_A2_vaddubs(0, 0);
  // CHECK: @llvm.hexagon.A2.vadduhs
  __builtin_HEXAGON_A2_vadduhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddw
  __builtin_HEXAGON_A2_vaddw(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddws
  __builtin_HEXAGON_A2_vaddws(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgh
  __builtin_HEXAGON_A2_vavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.vavghcr
  __builtin_HEXAGON_A2_vavghcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavghr
  __builtin_HEXAGON_A2_vavghr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgub
  __builtin_HEXAGON_A2_vavgub(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgubr
  __builtin_HEXAGON_A2_vavgubr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguh
  __builtin_HEXAGON_A2_vavguh(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguhr
  __builtin_HEXAGON_A2_vavguhr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguw
  __builtin_HEXAGON_A2_vavguw(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguwr
  __builtin_HEXAGON_A2_vavguwr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgw
  __builtin_HEXAGON_A2_vavgw(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgwcr
  __builtin_HEXAGON_A2_vavgwcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgwr
  __builtin_HEXAGON_A2_vavgwr(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpbeq
  __builtin_HEXAGON_A2_vcmpbeq(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpbgtu
  __builtin_HEXAGON_A2_vcmpbgtu(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpheq
  __builtin_HEXAGON_A2_vcmpheq(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmphgt
  __builtin_HEXAGON_A2_vcmphgt(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmphgtu
  __builtin_HEXAGON_A2_vcmphgtu(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpweq
  __builtin_HEXAGON_A2_vcmpweq(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpwgt
  __builtin_HEXAGON_A2_vcmpwgt(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpwgtu
  __builtin_HEXAGON_A2_vcmpwgtu(0, 0);
  // CHECK: @llvm.hexagon.A2.vconj
  __builtin_HEXAGON_A2_vconj(0);
  // CHECK: @llvm.hexagon.A2.vmaxb
  __builtin_HEXAGON_A2_vmaxb(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxh
  __builtin_HEXAGON_A2_vmaxh(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxub
  __builtin_HEXAGON_A2_vmaxub(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxuh
  __builtin_HEXAGON_A2_vmaxuh(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxuw
  __builtin_HEXAGON_A2_vmaxuw(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxw
  __builtin_HEXAGON_A2_vmaxw(0, 0);
  // CHECK: @llvm.hexagon.A2.vminb
  __builtin_HEXAGON_A2_vminb(0, 0);
  // CHECK: @llvm.hexagon.A2.vminh
  __builtin_HEXAGON_A2_vminh(0, 0);
  // CHECK: @llvm.hexagon.A2.vminub
  __builtin_HEXAGON_A2_vminub(0, 0);
  // CHECK: @llvm.hexagon.A2.vminuh
  __builtin_HEXAGON_A2_vminuh(0, 0);
  // CHECK: @llvm.hexagon.A2.vminuw
  __builtin_HEXAGON_A2_vminuw(0, 0);
  // CHECK: @llvm.hexagon.A2.vminw
  __builtin_HEXAGON_A2_vminw(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgh
  __builtin_HEXAGON_A2_vnavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavghcr
  __builtin_HEXAGON_A2_vnavghcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavghr
  __builtin_HEXAGON_A2_vnavghr(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgw
  __builtin_HEXAGON_A2_vnavgw(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgwcr
  __builtin_HEXAGON_A2_vnavgwcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgwr
  __builtin_HEXAGON_A2_vnavgwr(0, 0);
  // CHECK: @llvm.hexagon.A2.vraddub
  __builtin_HEXAGON_A2_vraddub(0, 0);
  // CHECK: @llvm.hexagon.A2.vraddub.acc
  __builtin_HEXAGON_A2_vraddub_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.A2.vrsadub
  __builtin_HEXAGON_A2_vrsadub(0, 0);
  // CHECK: @llvm.hexagon.A2.vrsadub.acc
  __builtin_HEXAGON_A2_vrsadub_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.A2.vsubb.map
  __builtin_HEXAGON_A2_vsubb_map(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubh
  __builtin_HEXAGON_A2_vsubh(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubhs
  __builtin_HEXAGON_A2_vsubhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubub
  __builtin_HEXAGON_A2_vsubub(0, 0);
  // CHECK: @llvm.hexagon.A2.vsububs
  __builtin_HEXAGON_A2_vsububs(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubuhs
  __builtin_HEXAGON_A2_vsubuhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubw
  __builtin_HEXAGON_A2_vsubw(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubws
  __builtin_HEXAGON_A2_vsubws(0, 0);
  // CHECK: @llvm.hexagon.A2.xor
  __builtin_HEXAGON_A2_xor(0, 0);
  // CHECK: @llvm.hexagon.A2.xorp
  __builtin_HEXAGON_A2_xorp(0, 0);
  // CHECK: @llvm.hexagon.A2.zxtb
  __builtin_HEXAGON_A2_zxtb(0);
  // CHECK: @llvm.hexagon.A2.zxth
  __builtin_HEXAGON_A2_zxth(0);
  // CHECK: @llvm.hexagon.A4.andn
  __builtin_HEXAGON_A4_andn(0, 0);
  // CHECK: @llvm.hexagon.A4.andnp
  __builtin_HEXAGON_A4_andnp(0, 0);
  // CHECK: @llvm.hexagon.A4.bitsplit
  __builtin_HEXAGON_A4_bitsplit(0, 0);
  // CHECK: @llvm.hexagon.A4.bitspliti
  __builtin_HEXAGON_A4_bitspliti(0, 0);
  // CHECK: @llvm.hexagon.A4.boundscheck
  __builtin_HEXAGON_A4_boundscheck(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbeq
  __builtin_HEXAGON_A4_cmpbeq(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbeqi
  __builtin_HEXAGON_A4_cmpbeqi(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgt
  __builtin_HEXAGON_A4_cmpbgt(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgti
  __builtin_HEXAGON_A4_cmpbgti(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgtu
  __builtin_HEXAGON_A4_cmpbgtu(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgtui
  __builtin_HEXAGON_A4_cmpbgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpheq
  __builtin_HEXAGON_A4_cmpheq(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpheqi
  __builtin_HEXAGON_A4_cmpheqi(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgt
  __builtin_HEXAGON_A4_cmphgt(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgti
  __builtin_HEXAGON_A4_cmphgti(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgtu
  __builtin_HEXAGON_A4_cmphgtu(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgtui
  __builtin_HEXAGON_A4_cmphgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.combineir
  __builtin_HEXAGON_A4_combineir(0, 0);
  // CHECK: @llvm.hexagon.A4.combineri
  __builtin_HEXAGON_A4_combineri(0, 0);
  // CHECK: @llvm.hexagon.A4.cround.ri
  __builtin_HEXAGON_A4_cround_ri(0, 0);
  // CHECK: @llvm.hexagon.A4.cround.rr
  __builtin_HEXAGON_A4_cround_rr(0, 0);
  // CHECK: @llvm.hexagon.A4.modwrapu
  __builtin_HEXAGON_A4_modwrapu(0, 0);
  // CHECK: @llvm.hexagon.A4.orn
  __builtin_HEXAGON_A4_orn(0, 0);
  // CHECK: @llvm.hexagon.A4.ornp
  __builtin_HEXAGON_A4_ornp(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpeq
  __builtin_HEXAGON_A4_rcmpeq(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpeqi
  __builtin_HEXAGON_A4_rcmpeqi(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpneq
  __builtin_HEXAGON_A4_rcmpneq(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpneqi
  __builtin_HEXAGON_A4_rcmpneqi(0, 0);
  // CHECK: @llvm.hexagon.A4.round.ri
  __builtin_HEXAGON_A4_round_ri(0, 0);
  // CHECK: @llvm.hexagon.A4.round.ri.sat
  __builtin_HEXAGON_A4_round_ri_sat(0, 0);
  // CHECK: @llvm.hexagon.A4.round.rr
  __builtin_HEXAGON_A4_round_rr(0, 0);
  // CHECK: @llvm.hexagon.A4.round.rr.sat
  __builtin_HEXAGON_A4_round_rr_sat(0, 0);
  // CHECK: @llvm.hexagon.A4.tlbmatch
  __builtin_HEXAGON_A4_tlbmatch(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbeq.any
  __builtin_HEXAGON_A4_vcmpbeq_any(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbeqi
  __builtin_HEXAGON_A4_vcmpbeqi(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbgt
  __builtin_HEXAGON_A4_vcmpbgt(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbgti
  __builtin_HEXAGON_A4_vcmpbgti(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbgtui
  __builtin_HEXAGON_A4_vcmpbgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpheqi
  __builtin_HEXAGON_A4_vcmpheqi(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmphgti
  __builtin_HEXAGON_A4_vcmphgti(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmphgtui
  __builtin_HEXAGON_A4_vcmphgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpweqi
  __builtin_HEXAGON_A4_vcmpweqi(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpwgti
  __builtin_HEXAGON_A4_vcmpwgti(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpwgtui
  __builtin_HEXAGON_A4_vcmpwgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxh
  __builtin_HEXAGON_A4_vrmaxh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxuh
  __builtin_HEXAGON_A4_vrmaxuh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxuw
  __builtin_HEXAGON_A4_vrmaxuw(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxw
  __builtin_HEXAGON_A4_vrmaxw(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminh
  __builtin_HEXAGON_A4_vrminh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminuh
  __builtin_HEXAGON_A4_vrminuh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminuw
  __builtin_HEXAGON_A4_vrminuw(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminw
  __builtin_HEXAGON_A4_vrminw(0, 0, 0);
  // CHECK: @llvm.hexagon.A5.vaddhubs
  __builtin_HEXAGON_A5_vaddhubs(0, 0);
  // CHECK: @llvm.hexagon.A6.vcmpbeq.notany
  __builtin_HEXAGON_A6_vcmpbeq_notany(0, 0);
  // CHECK: @llvm.hexagon.A6.vcmpbeq.notany.128B
  __builtin_HEXAGON_A6_vcmpbeq_notany_128B(0, 0);
  // CHECK: @llvm.hexagon.C2.all8
  __builtin_HEXAGON_C2_all8(0);
  // CHECK: @llvm.hexagon.C2.and
  __builtin_HEXAGON_C2_and(0, 0);
  // CHECK: @llvm.hexagon.C2.andn
  __builtin_HEXAGON_C2_andn(0, 0);
  // CHECK: @llvm.hexagon.C2.any8
  __builtin_HEXAGON_C2_any8(0);
  // CHECK: @llvm.hexagon.C2.bitsclr
  __builtin_HEXAGON_C2_bitsclr(0, 0);
  // CHECK: @llvm.hexagon.C2.bitsclri
  __builtin_HEXAGON_C2_bitsclri(0, 0);
  // CHECK: @llvm.hexagon.C2.bitsset
  __builtin_HEXAGON_C2_bitsset(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpeq
  __builtin_HEXAGON_C2_cmpeq(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpeqi
  __builtin_HEXAGON_C2_cmpeqi(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpeqp
  __builtin_HEXAGON_C2_cmpeqp(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgei
  __builtin_HEXAGON_C2_cmpgei(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgeui
  __builtin_HEXAGON_C2_cmpgeui(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgt
  __builtin_HEXAGON_C2_cmpgt(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgti
  __builtin_HEXAGON_C2_cmpgti(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtp
  __builtin_HEXAGON_C2_cmpgtp(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtu
  __builtin_HEXAGON_C2_cmpgtu(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtui
  __builtin_HEXAGON_C2_cmpgtui(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtup
  __builtin_HEXAGON_C2_cmpgtup(0, 0);
  // CHECK: @llvm.hexagon.C2.cmplt
  __builtin_HEXAGON_C2_cmplt(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpltu
  __builtin_HEXAGON_C2_cmpltu(0, 0);
  // CHECK: @llvm.hexagon.C2.mask
  __builtin_HEXAGON_C2_mask(0);
  // CHECK: @llvm.hexagon.C2.mux
  __builtin_HEXAGON_C2_mux(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.muxii
  __builtin_HEXAGON_C2_muxii(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.muxir
  __builtin_HEXAGON_C2_muxir(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.muxri
  __builtin_HEXAGON_C2_muxri(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.not
  __builtin_HEXAGON_C2_not(0);
  // CHECK: @llvm.hexagon.C2.or
  __builtin_HEXAGON_C2_or(0, 0);
  // CHECK: @llvm.hexagon.C2.orn
  __builtin_HEXAGON_C2_orn(0, 0);
  // CHECK: @llvm.hexagon.C2.pxfer.map
  __builtin_HEXAGON_C2_pxfer_map(0);
  // CHECK: @llvm.hexagon.C2.tfrpr
  __builtin_HEXAGON_C2_tfrpr(0);
  // CHECK: @llvm.hexagon.C2.tfrrp
  __builtin_HEXAGON_C2_tfrrp(0);
  // CHECK: @llvm.hexagon.C2.vitpack
  __builtin_HEXAGON_C2_vitpack(0, 0);
  // CHECK: @llvm.hexagon.C2.vmux
  __builtin_HEXAGON_C2_vmux(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.xor
  __builtin_HEXAGON_C2_xor(0, 0);
  // CHECK: @llvm.hexagon.C4.and.and
  __builtin_HEXAGON_C4_and_and(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.and.andn
  __builtin_HEXAGON_C4_and_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.and.or
  __builtin_HEXAGON_C4_and_or(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.and.orn
  __builtin_HEXAGON_C4_and_orn(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.cmplte
  __builtin_HEXAGON_C4_cmplte(0, 0);
  // CHECK: @llvm.hexagon.C4.cmpltei
  __builtin_HEXAGON_C4_cmpltei(0, 0);
  // CHECK: @llvm.hexagon.C4.cmplteu
  __builtin_HEXAGON_C4_cmplteu(0, 0);
  // CHECK: @llvm.hexagon.C4.cmplteui
  __builtin_HEXAGON_C4_cmplteui(0, 0);
  // CHECK: @llvm.hexagon.C4.cmpneq
  __builtin_HEXAGON_C4_cmpneq(0, 0);
  // CHECK: @llvm.hexagon.C4.cmpneqi
  __builtin_HEXAGON_C4_cmpneqi(0, 0);
  // CHECK: @llvm.hexagon.C4.fastcorner9
  __builtin_HEXAGON_C4_fastcorner9(0, 0);
  // CHECK: @llvm.hexagon.C4.fastcorner9.not
  __builtin_HEXAGON_C4_fastcorner9_not(0, 0);
  // CHECK: @llvm.hexagon.C4.nbitsclr
  __builtin_HEXAGON_C4_nbitsclr(0, 0);
  // CHECK: @llvm.hexagon.C4.nbitsclri
  __builtin_HEXAGON_C4_nbitsclri(0, 0);
  // CHECK: @llvm.hexagon.C4.nbitsset
  __builtin_HEXAGON_C4_nbitsset(0, 0);
  // CHECK: @llvm.hexagon.C4.or.and
  __builtin_HEXAGON_C4_or_and(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.or.andn
  __builtin_HEXAGON_C4_or_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.or.or
  __builtin_HEXAGON_C4_or_or(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.or.orn
  __builtin_HEXAGON_C4_or_orn(0, 0, 0);
  // CHECK: @llvm.hexagon.F2.conv.d2df
  __builtin_HEXAGON_F2_conv_d2df(0);
  // CHECK: @llvm.hexagon.F2.conv.d2sf
  __builtin_HEXAGON_F2_conv_d2sf(0);
  // CHECK: @llvm.hexagon.F2.conv.df2d
  __builtin_HEXAGON_F2_conv_df2d(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2d.chop
  __builtin_HEXAGON_F2_conv_df2d_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2sf
  __builtin_HEXAGON_F2_conv_df2sf(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2ud
  __builtin_HEXAGON_F2_conv_df2ud(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2ud.chop
  __builtin_HEXAGON_F2_conv_df2ud_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2uw
  __builtin_HEXAGON_F2_conv_df2uw(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2uw.chop
  __builtin_HEXAGON_F2_conv_df2uw_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2w
  __builtin_HEXAGON_F2_conv_df2w(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2w.chop
  __builtin_HEXAGON_F2_conv_df2w_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.sf2d
  __builtin_HEXAGON_F2_conv_sf2d(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2d.chop
  __builtin_HEXAGON_F2_conv_sf2d_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2df
  __builtin_HEXAGON_F2_conv_sf2df(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2ud
  __builtin_HEXAGON_F2_conv_sf2ud(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2ud.chop
  __builtin_HEXAGON_F2_conv_sf2ud_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2uw
  __builtin_HEXAGON_F2_conv_sf2uw(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2uw.chop
  __builtin_HEXAGON_F2_conv_sf2uw_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2w
  __builtin_HEXAGON_F2_conv_sf2w(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2w.chop
  __builtin_HEXAGON_F2_conv_sf2w_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.ud2df
  __builtin_HEXAGON_F2_conv_ud2df(0);
  // CHECK: @llvm.hexagon.F2.conv.ud2sf
  __builtin_HEXAGON_F2_conv_ud2sf(0);
  // CHECK: @llvm.hexagon.F2.conv.uw2df
  __builtin_HEXAGON_F2_conv_uw2df(0);
  // CHECK: @llvm.hexagon.F2.conv.uw2sf
  __builtin_HEXAGON_F2_conv_uw2sf(0);
  // CHECK: @llvm.hexagon.F2.conv.w2df
  __builtin_HEXAGON_F2_conv_w2df(0);
  // CHECK: @llvm.hexagon.F2.conv.w2sf
  __builtin_HEXAGON_F2_conv_w2sf(0);
  // CHECK: @llvm.hexagon.F2.dfclass
  __builtin_HEXAGON_F2_dfclass(0.0, 0);
  // CHECK: @llvm.hexagon.F2.dfcmpeq
  __builtin_HEXAGON_F2_dfcmpeq(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfcmpge
  __builtin_HEXAGON_F2_dfcmpge(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfcmpgt
  __builtin_HEXAGON_F2_dfcmpgt(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfcmpuo
  __builtin_HEXAGON_F2_dfcmpuo(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfimm.n
  __builtin_HEXAGON_F2_dfimm_n(0);
  // CHECK: @llvm.hexagon.F2.dfimm.p
  __builtin_HEXAGON_F2_dfimm_p(0);
  // CHECK: @llvm.hexagon.F2.sfadd
  __builtin_HEXAGON_F2_sfadd(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfclass
  __builtin_HEXAGON_F2_sfclass(0.0f, 0);
  // CHECK: @llvm.hexagon.F2.sfcmpeq
  __builtin_HEXAGON_F2_sfcmpeq(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfcmpge
  __builtin_HEXAGON_F2_sfcmpge(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfcmpgt
  __builtin_HEXAGON_F2_sfcmpgt(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfcmpuo
  __builtin_HEXAGON_F2_sfcmpuo(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffixupd
  __builtin_HEXAGON_F2_sffixupd(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffixupn
  __builtin_HEXAGON_F2_sffixupn(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffixupr
  __builtin_HEXAGON_F2_sffixupr(0.0f);
  // CHECK: @llvm.hexagon.F2.sffma
  __builtin_HEXAGON_F2_sffma(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffma.lib
  __builtin_HEXAGON_F2_sffma_lib(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffma.sc
  __builtin_HEXAGON_F2_sffma_sc(0.0f, 0.0f, 0.0f, 0);
  // CHECK: @llvm.hexagon.F2.sffms
  __builtin_HEXAGON_F2_sffms(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffms.lib
  __builtin_HEXAGON_F2_sffms_lib(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfimm.n
  __builtin_HEXAGON_F2_sfimm_n(0);
  // CHECK: @llvm.hexagon.F2.sfimm.p
  __builtin_HEXAGON_F2_sfimm_p(0);
  // CHECK: @llvm.hexagon.F2.sfmax
  __builtin_HEXAGON_F2_sfmax(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfmin
  __builtin_HEXAGON_F2_sfmin(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfmpy
  __builtin_HEXAGON_F2_sfmpy(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfsub
  __builtin_HEXAGON_F2_sfsub(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.M2.acci
  __builtin_HEXAGON_M2_acci(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.accii
  __builtin_HEXAGON_M2_accii(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmaci.s0
  __builtin_HEXAGON_M2_cmaci_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacr.s0
  __builtin_HEXAGON_M2_cmacr_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacs.s0
  __builtin_HEXAGON_M2_cmacs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacs.s1
  __builtin_HEXAGON_M2_cmacs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacsc.s0
  __builtin_HEXAGON_M2_cmacsc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacsc.s1
  __builtin_HEXAGON_M2_cmacsc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyi.s0
  __builtin_HEXAGON_M2_cmpyi_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyr.s0
  __builtin_HEXAGON_M2_cmpyr_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrs.s0
  __builtin_HEXAGON_M2_cmpyrs_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrs.s1
  __builtin_HEXAGON_M2_cmpyrs_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrsc.s0
  __builtin_HEXAGON_M2_cmpyrsc_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrsc.s1
  __builtin_HEXAGON_M2_cmpyrsc_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpys.s0
  __builtin_HEXAGON_M2_cmpys_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpys.s1
  __builtin_HEXAGON_M2_cmpys_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpysc.s0
  __builtin_HEXAGON_M2_cmpysc_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpysc.s1
  __builtin_HEXAGON_M2_cmpysc_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cnacs.s0
  __builtin_HEXAGON_M2_cnacs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cnacs.s1
  __builtin_HEXAGON_M2_cnacs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cnacsc.s0
  __builtin_HEXAGON_M2_cnacsc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cnacsc.s1
  __builtin_HEXAGON_M2_cnacsc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.acc.s0
  __builtin_HEXAGON_M2_dpmpyss_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.nac.s0
  __builtin_HEXAGON_M2_dpmpyss_nac_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.rnd.s0
  __builtin_HEXAGON_M2_dpmpyss_rnd_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.s0
  __builtin_HEXAGON_M2_dpmpyss_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyuu.acc.s0
  __builtin_HEXAGON_M2_dpmpyuu_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyuu.nac.s0
  __builtin_HEXAGON_M2_dpmpyuu_nac_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyuu.s0
  __builtin_HEXAGON_M2_dpmpyuu_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyh.rs1
  __builtin_HEXAGON_M2_hmmpyh_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyh.s1
  __builtin_HEXAGON_M2_hmmpyh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyl.rs1
  __builtin_HEXAGON_M2_hmmpyl_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyl.s1
  __builtin_HEXAGON_M2_hmmpyl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.maci
  __builtin_HEXAGON_M2_maci(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.macsin
  __builtin_HEXAGON_M2_macsin(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.macsip
  __builtin_HEXAGON_M2_macsip(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.rs0
  __builtin_HEXAGON_M2_mmachs_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.rs1
  __builtin_HEXAGON_M2_mmachs_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.s0
  __builtin_HEXAGON_M2_mmachs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.s1
  __builtin_HEXAGON_M2_mmachs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.rs0
  __builtin_HEXAGON_M2_mmacls_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.rs1
  __builtin_HEXAGON_M2_mmacls_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.s0
  __builtin_HEXAGON_M2_mmacls_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.s1
  __builtin_HEXAGON_M2_mmacls_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.rs0
  __builtin_HEXAGON_M2_mmacuhs_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.rs1
  __builtin_HEXAGON_M2_mmacuhs_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.s0
  __builtin_HEXAGON_M2_mmacuhs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.s1
  __builtin_HEXAGON_M2_mmacuhs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.rs0
  __builtin_HEXAGON_M2_mmaculs_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.rs1
  __builtin_HEXAGON_M2_mmaculs_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.s0
  __builtin_HEXAGON_M2_mmaculs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.s1
  __builtin_HEXAGON_M2_mmaculs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.rs0
  __builtin_HEXAGON_M2_mmpyh_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.rs1
  __builtin_HEXAGON_M2_mmpyh_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.s0
  __builtin_HEXAGON_M2_mmpyh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.s1
  __builtin_HEXAGON_M2_mmpyh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.rs0
  __builtin_HEXAGON_M2_mmpyl_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.rs1
  __builtin_HEXAGON_M2_mmpyl_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.s0
  __builtin_HEXAGON_M2_mmpyl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.s1
  __builtin_HEXAGON_M2_mmpyl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.rs0
  __builtin_HEXAGON_M2_mmpyuh_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.rs1
  __builtin_HEXAGON_M2_mmpyuh_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.s0
  __builtin_HEXAGON_M2_mmpyuh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.s1
  __builtin_HEXAGON_M2_mmpyuh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.rs0
  __builtin_HEXAGON_M2_mmpyul_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.rs1
  __builtin_HEXAGON_M2_mmpyul_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.s0
  __builtin_HEXAGON_M2_mmpyul_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.s1
  __builtin_HEXAGON_M2_mmpyul_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hh.s0
  __builtin_HEXAGON_M2_mpy_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hh.s1
  __builtin_HEXAGON_M2_mpy_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hl.s0
  __builtin_HEXAGON_M2_mpy_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hl.s1
  __builtin_HEXAGON_M2_mpy_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.lh.s0
  __builtin_HEXAGON_M2_mpy_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.lh.s1
  __builtin_HEXAGON_M2_mpy_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.ll.s0
  __builtin_HEXAGON_M2_mpy_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.ll.s1
  __builtin_HEXAGON_M2_mpy_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hh.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hh.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hl.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hl.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.lh.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.lh.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.ll.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.ll.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hh.s0
  __builtin_HEXAGON_M2_mpy_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hh.s1
  __builtin_HEXAGON_M2_mpy_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hl.s0
  __builtin_HEXAGON_M2_mpy_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hl.s1
  __builtin_HEXAGON_M2_mpy_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.lh.s0
  __builtin_HEXAGON_M2_mpy_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.lh.s1
  __builtin_HEXAGON_M2_mpy_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.ll.s0
  __builtin_HEXAGON_M2_mpy_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.ll.s1
  __builtin_HEXAGON_M2_mpy_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hh.s0
  __builtin_HEXAGON_M2_mpy_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hh.s1
  __builtin_HEXAGON_M2_mpy_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hl.s0
  __builtin_HEXAGON_M2_mpy_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hl.s1
  __builtin_HEXAGON_M2_mpy_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.lh.s0
  __builtin_HEXAGON_M2_mpy_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.lh.s1
  __builtin_HEXAGON_M2_mpy_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.ll.s0
  __builtin_HEXAGON_M2_mpy_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.ll.s1
  __builtin_HEXAGON_M2_mpy_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hh.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hh.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hl.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hl.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.lh.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.lh.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.ll.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.ll.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hh.s0
  __builtin_HEXAGON_M2_mpy_rnd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hh.s1
  __builtin_HEXAGON_M2_mpy_rnd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hl.s0
  __builtin_HEXAGON_M2_mpy_rnd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hl.s1
  __builtin_HEXAGON_M2_mpy_rnd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.lh.s0
  __builtin_HEXAGON_M2_mpy_rnd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.lh.s1
  __builtin_HEXAGON_M2_mpy_rnd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.ll.s0
  __builtin_HEXAGON_M2_mpy_rnd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.ll.s1
  __builtin_HEXAGON_M2_mpy_rnd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hh.s0
  __builtin_HEXAGON_M2_mpy_sat_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hh.s1
  __builtin_HEXAGON_M2_mpy_sat_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hl.s0
  __builtin_HEXAGON_M2_mpy_sat_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hl.s1
  __builtin_HEXAGON_M2_mpy_sat_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.lh.s0
  __builtin_HEXAGON_M2_mpy_sat_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.lh.s1
  __builtin_HEXAGON_M2_mpy_sat_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.ll.s0
  __builtin_HEXAGON_M2_mpy_sat_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.ll.s1
  __builtin_HEXAGON_M2_mpy_sat_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hh.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hh.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hl.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hl.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.lh.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.lh.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.ll.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.ll.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.up
  __builtin_HEXAGON_M2_mpy_up(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.up.s1
  __builtin_HEXAGON_M2_mpy_up_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.up.s1.sat
  __builtin_HEXAGON_M2_mpy_up_s1_sat(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hh.s0
  __builtin_HEXAGON_M2_mpyd_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hh.s1
  __builtin_HEXAGON_M2_mpyd_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hl.s0
  __builtin_HEXAGON_M2_mpyd_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hl.s1
  __builtin_HEXAGON_M2_mpyd_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.lh.s0
  __builtin_HEXAGON_M2_mpyd_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.lh.s1
  __builtin_HEXAGON_M2_mpyd_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.ll.s0
  __builtin_HEXAGON_M2_mpyd_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.ll.s1
  __builtin_HEXAGON_M2_mpyd_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hh.s0
  __builtin_HEXAGON_M2_mpyd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hh.s1
  __builtin_HEXAGON_M2_mpyd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hl.s0
  __builtin_HEXAGON_M2_mpyd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hl.s1
  __builtin_HEXAGON_M2_mpyd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.lh.s0
  __builtin_HEXAGON_M2_mpyd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.lh.s1
  __builtin_HEXAGON_M2_mpyd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.ll.s0
  __builtin_HEXAGON_M2_mpyd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.ll.s1
  __builtin_HEXAGON_M2_mpyd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hh.s0
  __builtin_HEXAGON_M2_mpyd_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hh.s1
  __builtin_HEXAGON_M2_mpyd_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hl.s0
  __builtin_HEXAGON_M2_mpyd_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hl.s1
  __builtin_HEXAGON_M2_mpyd_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.lh.s0
  __builtin_HEXAGON_M2_mpyd_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.lh.s1
  __builtin_HEXAGON_M2_mpyd_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.ll.s0
  __builtin_HEXAGON_M2_mpyd_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.ll.s1
  __builtin_HEXAGON_M2_mpyd_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hh.s0
  __builtin_HEXAGON_M2_mpyd_rnd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hh.s1
  __builtin_HEXAGON_M2_mpyd_rnd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hl.s0
  __builtin_HEXAGON_M2_mpyd_rnd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hl.s1
  __builtin_HEXAGON_M2_mpyd_rnd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.lh.s0
  __builtin_HEXAGON_M2_mpyd_rnd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.lh.s1
  __builtin_HEXAGON_M2_mpyd_rnd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.ll.s0
  __builtin_HEXAGON_M2_mpyd_rnd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.ll.s1
  __builtin_HEXAGON_M2_mpyd_rnd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyi
  __builtin_HEXAGON_M2_mpyi(0, 0);
  // CHECK: @llvm.hexagon.M2.mpysmi
  __builtin_HEXAGON_M2_mpysmi(0, 0);
  // CHECK: @llvm.hexagon.M2.mpysu.up
  __builtin_HEXAGON_M2_mpysu_up(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hh.s0
  __builtin_HEXAGON_M2_mpyu_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hh.s1
  __builtin_HEXAGON_M2_mpyu_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hl.s0
  __builtin_HEXAGON_M2_mpyu_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hl.s1
  __builtin_HEXAGON_M2_mpyu_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.lh.s0
  __builtin_HEXAGON_M2_mpyu_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.lh.s1
  __builtin_HEXAGON_M2_mpyu_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.ll.s0
  __builtin_HEXAGON_M2_mpyu_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.ll.s1
  __builtin_HEXAGON_M2_mpyu_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hh.s0
  __builtin_HEXAGON_M2_mpyu_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hh.s1
  __builtin_HEXAGON_M2_mpyu_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hl.s0
  __builtin_HEXAGON_M2_mpyu_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hl.s1
  __builtin_HEXAGON_M2_mpyu_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.lh.s0
  __builtin_HEXAGON_M2_mpyu_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.lh.s1
  __builtin_HEXAGON_M2_mpyu_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.ll.s0
  __builtin_HEXAGON_M2_mpyu_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.ll.s1
  __builtin_HEXAGON_M2_mpyu_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hh.s0
  __builtin_HEXAGON_M2_mpyu_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hh.s1
  __builtin_HEXAGON_M2_mpyu_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hl.s0
  __builtin_HEXAGON_M2_mpyu_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hl.s1
  __builtin_HEXAGON_M2_mpyu_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.lh.s0
  __builtin_HEXAGON_M2_mpyu_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.lh.s1
  __builtin_HEXAGON_M2_mpyu_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.ll.s0
  __builtin_HEXAGON_M2_mpyu_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.ll.s1
  __builtin_HEXAGON_M2_mpyu_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.up
  __builtin_HEXAGON_M2_mpyu_up(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hh.s0
  __builtin_HEXAGON_M2_mpyud_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hh.s1
  __builtin_HEXAGON_M2_mpyud_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hl.s0
  __builtin_HEXAGON_M2_mpyud_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hl.s1
  __builtin_HEXAGON_M2_mpyud_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.lh.s0
  __builtin_HEXAGON_M2_mpyud_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.lh.s1
  __builtin_HEXAGON_M2_mpyud_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.ll.s0
  __builtin_HEXAGON_M2_mpyud_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.ll.s1
  __builtin_HEXAGON_M2_mpyud_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hh.s0
  __builtin_HEXAGON_M2_mpyud_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hh.s1
  __builtin_HEXAGON_M2_mpyud_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hl.s0
  __builtin_HEXAGON_M2_mpyud_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hl.s1
  __builtin_HEXAGON_M2_mpyud_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.lh.s0
  __builtin_HEXAGON_M2_mpyud_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.lh.s1
  __builtin_HEXAGON_M2_mpyud_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.ll.s0
  __builtin_HEXAGON_M2_mpyud_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.ll.s1
  __builtin_HEXAGON_M2_mpyud_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hh.s0
  __builtin_HEXAGON_M2_mpyud_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hh.s1
  __builtin_HEXAGON_M2_mpyud_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hl.s0
  __builtin_HEXAGON_M2_mpyud_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hl.s1
  __builtin_HEXAGON_M2_mpyud_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.lh.s0
  __builtin_HEXAGON_M2_mpyud_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.lh.s1
  __builtin_HEXAGON_M2_mpyud_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.ll.s0
  __builtin_HEXAGON_M2_mpyud_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.ll.s1
  __builtin_HEXAGON_M2_mpyud_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyui
  __builtin_HEXAGON_M2_mpyui(0, 0);
  // CHECK: @llvm.hexagon.M2.nacci
  __builtin_HEXAGON_M2_nacci(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.naccii
  __builtin_HEXAGON_M2_naccii(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.subacc
  __builtin_HEXAGON_M2_subacc(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vabsdiffh
  __builtin_HEXAGON_M2_vabsdiffh(0, 0);
  // CHECK: @llvm.hexagon.M2.vabsdiffw
  __builtin_HEXAGON_M2_vabsdiffw(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmac.s0.sat.i
  __builtin_HEXAGON_M2_vcmac_s0_sat_i(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vcmac.s0.sat.r
  __builtin_HEXAGON_M2_vcmac_s0_sat_r(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s0.sat.i
  __builtin_HEXAGON_M2_vcmpy_s0_sat_i(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s0.sat.r
  __builtin_HEXAGON_M2_vcmpy_s0_sat_r(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s1.sat.i
  __builtin_HEXAGON_M2_vcmpy_s1_sat_i(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s1.sat.r
  __builtin_HEXAGON_M2_vcmpy_s1_sat_r(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmacs.s0
  __builtin_HEXAGON_M2_vdmacs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vdmacs.s1
  __builtin_HEXAGON_M2_vdmacs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpyrs.s0
  __builtin_HEXAGON_M2_vdmpyrs_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpyrs.s1
  __builtin_HEXAGON_M2_vdmpyrs_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpys.s0
  __builtin_HEXAGON_M2_vdmpys_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpys.s1
  __builtin_HEXAGON_M2_vdmpys_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2
  __builtin_HEXAGON_M2_vmac2(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2es
  __builtin_HEXAGON_M2_vmac2es(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2es.s0
  __builtin_HEXAGON_M2_vmac2es_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2es.s1
  __builtin_HEXAGON_M2_vmac2es_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2s.s0
  __builtin_HEXAGON_M2_vmac2s_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2s.s1
  __builtin_HEXAGON_M2_vmac2s_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2su.s0
  __builtin_HEXAGON_M2_vmac2su_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2su.s1
  __builtin_HEXAGON_M2_vmac2su_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2es.s0
  __builtin_HEXAGON_M2_vmpy2es_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2es.s1
  __builtin_HEXAGON_M2_vmpy2es_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s0
  __builtin_HEXAGON_M2_vmpy2s_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s0pack
  __builtin_HEXAGON_M2_vmpy2s_s0pack(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s1
  __builtin_HEXAGON_M2_vmpy2s_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s1pack
  __builtin_HEXAGON_M2_vmpy2s_s1pack(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2su.s0
  __builtin_HEXAGON_M2_vmpy2su_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2su.s1
  __builtin_HEXAGON_M2_vmpy2su_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vraddh
  __builtin_HEXAGON_M2_vraddh(0, 0);
  // CHECK: @llvm.hexagon.M2.vradduh
  __builtin_HEXAGON_M2_vradduh(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmaci.s0
  __builtin_HEXAGON_M2_vrcmaci_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmaci.s0c
  __builtin_HEXAGON_M2_vrcmaci_s0c(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmacr.s0
  __builtin_HEXAGON_M2_vrcmacr_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmacr.s0c
  __builtin_HEXAGON_M2_vrcmacr_s0c(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyi.s0
  __builtin_HEXAGON_M2_vrcmpyi_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyi.s0c
  __builtin_HEXAGON_M2_vrcmpyi_s0c(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyr.s0
  __builtin_HEXAGON_M2_vrcmpyr_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyr.s0c
  __builtin_HEXAGON_M2_vrcmpyr_s0c(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpys.acc.s1
  __builtin_HEXAGON_M2_vrcmpys_acc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpys.s1
  __builtin_HEXAGON_M2_vrcmpys_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpys.s1rp
  __builtin_HEXAGON_M2_vrcmpys_s1rp(0, 0);
  // CHECK: @llvm.hexagon.M2.vrmac.s0
  __builtin_HEXAGON_M2_vrmac_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrmpy.s0
  __builtin_HEXAGON_M2_vrmpy_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.xor.xacc
  __builtin_HEXAGON_M2_xor_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.and
  __builtin_HEXAGON_M4_and_and(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.andn
  __builtin_HEXAGON_M4_and_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.or
  __builtin_HEXAGON_M4_and_or(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.xor
  __builtin_HEXAGON_M4_and_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyi.wh
  __builtin_HEXAGON_M4_cmpyi_wh(0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyi.whc
  __builtin_HEXAGON_M4_cmpyi_whc(0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyr.wh
  __builtin_HEXAGON_M4_cmpyr_wh(0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyr.whc
  __builtin_HEXAGON_M4_cmpyr_whc(0, 0);
  // CHECK: @llvm.hexagon.M4.mac.up.s1.sat
  __builtin_HEXAGON_M4_mac_up_s1_sat(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyri.addi
  __builtin_HEXAGON_M4_mpyri_addi(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyri.addr
  __builtin_HEXAGON_M4_mpyri_addr(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyri.addr.u2
  __builtin_HEXAGON_M4_mpyri_addr_u2(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyrr.addi
  __builtin_HEXAGON_M4_mpyrr_addi(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyrr.addr
  __builtin_HEXAGON_M4_mpyrr_addr(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.nac.up.s1.sat
  __builtin_HEXAGON_M4_nac_up_s1_sat(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.and
  __builtin_HEXAGON_M4_or_and(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.andn
  __builtin_HEXAGON_M4_or_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.or
  __builtin_HEXAGON_M4_or_or(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.xor
  __builtin_HEXAGON_M4_or_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.pmpyw
  __builtin_HEXAGON_M4_pmpyw(0, 0);
  // CHECK: @llvm.hexagon.M4.pmpyw.acc
  __builtin_HEXAGON_M4_pmpyw_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vpmpyh
  __builtin_HEXAGON_M4_vpmpyh(0, 0);
  // CHECK: @llvm.hexagon.M4.vpmpyh.acc
  __builtin_HEXAGON_M4_vpmpyh_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.acc.s0
  __builtin_HEXAGON_M4_vrmpyeh_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.acc.s1
  __builtin_HEXAGON_M4_vrmpyeh_acc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.s0
  __builtin_HEXAGON_M4_vrmpyeh_s0(0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.s1
  __builtin_HEXAGON_M4_vrmpyeh_s1(0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.acc.s0
  __builtin_HEXAGON_M4_vrmpyoh_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.acc.s1
  __builtin_HEXAGON_M4_vrmpyoh_acc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.s0
  __builtin_HEXAGON_M4_vrmpyoh_s0(0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.s1
  __builtin_HEXAGON_M4_vrmpyoh_s1(0, 0);
  // CHECK: @llvm.hexagon.M4.xor.and
  __builtin_HEXAGON_M4_xor_and(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.xor.andn
  __builtin_HEXAGON_M4_xor_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.xor.or
  __builtin_HEXAGON_M4_xor_or(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.xor.xacc
  __builtin_HEXAGON_M4_xor_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vdmacbsu
  __builtin_HEXAGON_M5_vdmacbsu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vdmpybsu
  __builtin_HEXAGON_M5_vdmpybsu(0, 0);
  // CHECK: @llvm.hexagon.M5.vmacbsu
  __builtin_HEXAGON_M5_vmacbsu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vmacbuu
  __builtin_HEXAGON_M5_vmacbuu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vmpybsu
  __builtin_HEXAGON_M5_vmpybsu(0, 0);
  // CHECK: @llvm.hexagon.M5.vmpybuu
  __builtin_HEXAGON_M5_vmpybuu(0, 0);
  // CHECK: @llvm.hexagon.M5.vrmacbsu
  __builtin_HEXAGON_M5_vrmacbsu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vrmacbuu
  __builtin_HEXAGON_M5_vrmacbuu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vrmpybsu
  __builtin_HEXAGON_M5_vrmpybsu(0, 0);
  // CHECK: @llvm.hexagon.M5.vrmpybuu
  __builtin_HEXAGON_M5_vrmpybuu(0, 0);
  // CHECK: @llvm.hexagon.M6.vabsdiffb
  __builtin_HEXAGON_M6_vabsdiffb(0, 0);
  // CHECK: @llvm.hexagon.M6.vabsdiffub
  __builtin_HEXAGON_M6_vabsdiffub(0, 0);
  // CHECK: @llvm.hexagon.S2.addasl.rrri
  __builtin_HEXAGON_S2_addasl_rrri(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p
  __builtin_HEXAGON_S2_asl_i_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.acc
  __builtin_HEXAGON_S2_asl_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.and
  __builtin_HEXAGON_S2_asl_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.nac
  __builtin_HEXAGON_S2_asl_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.or
  __builtin_HEXAGON_S2_asl_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.xacc
  __builtin_HEXAGON_S2_asl_i_p_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r
  __builtin_HEXAGON_S2_asl_i_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.acc
  __builtin_HEXAGON_S2_asl_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.and
  __builtin_HEXAGON_S2_asl_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.nac
  __builtin_HEXAGON_S2_asl_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.or
  __builtin_HEXAGON_S2_asl_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.sat
  __builtin_HEXAGON_S2_asl_i_r_sat(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.xacc
  __builtin_HEXAGON_S2_asl_i_r_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.vh
  __builtin_HEXAGON_S2_asl_i_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.vw
  __builtin_HEXAGON_S2_asl_i_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p
  __builtin_HEXAGON_S2_asl_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.acc
  __builtin_HEXAGON_S2_asl_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.and
  __builtin_HEXAGON_S2_asl_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.nac
  __builtin_HEXAGON_S2_asl_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.or
  __builtin_HEXAGON_S2_asl_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.xor
  __builtin_HEXAGON_S2_asl_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r
  __builtin_HEXAGON_S2_asl_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.acc
  __builtin_HEXAGON_S2_asl_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.and
  __builtin_HEXAGON_S2_asl_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.nac
  __builtin_HEXAGON_S2_asl_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.or
  __builtin_HEXAGON_S2_asl_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.sat
  __builtin_HEXAGON_S2_asl_r_r_sat(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.vh
  __builtin_HEXAGON_S2_asl_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.vw
  __builtin_HEXAGON_S2_asl_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p
  __builtin_HEXAGON_S2_asr_i_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.acc
  __builtin_HEXAGON_S2_asr_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.and
  __builtin_HEXAGON_S2_asr_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.nac
  __builtin_HEXAGON_S2_asr_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.or
  __builtin_HEXAGON_S2_asr_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.rnd
  __builtin_HEXAGON_S2_asr_i_p_rnd(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax
  __builtin_HEXAGON_S2_asr_i_p_rnd_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r
  __builtin_HEXAGON_S2_asr_i_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.acc
  __builtin_HEXAGON_S2_asr_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.and
  __builtin_HEXAGON_S2_asr_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.nac
  __builtin_HEXAGON_S2_asr_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.or
  __builtin_HEXAGON_S2_asr_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.rnd
  __builtin_HEXAGON_S2_asr_i_r_rnd(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.rnd.goodsyntax
  __builtin_HEXAGON_S2_asr_i_r_rnd_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.svw.trun
  __builtin_HEXAGON_S2_asr_i_svw_trun(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.vh
  __builtin_HEXAGON_S2_asr_i_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.vw
  __builtin_HEXAGON_S2_asr_i_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p
  __builtin_HEXAGON_S2_asr_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.acc
  __builtin_HEXAGON_S2_asr_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.and
  __builtin_HEXAGON_S2_asr_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.nac
  __builtin_HEXAGON_S2_asr_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.or
  __builtin_HEXAGON_S2_asr_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.xor
  __builtin_HEXAGON_S2_asr_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r
  __builtin_HEXAGON_S2_asr_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.acc
  __builtin_HEXAGON_S2_asr_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.and
  __builtin_HEXAGON_S2_asr_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.nac
  __builtin_HEXAGON_S2_asr_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.or
  __builtin_HEXAGON_S2_asr_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.sat
  __builtin_HEXAGON_S2_asr_r_r_sat(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.svw.trun
  __builtin_HEXAGON_S2_asr_r_svw_trun(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.vh
  __builtin_HEXAGON_S2_asr_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.vw
  __builtin_HEXAGON_S2_asr_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.brev
  __builtin_HEXAGON_S2_brev(0);
  // CHECK: @llvm.hexagon.S2.brevp
  __builtin_HEXAGON_S2_brevp(0);
  // CHECK: @llvm.hexagon.S2.cabacencbin
  __builtin_HEXAGON_S2_cabacencbin(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.cl0
  __builtin_HEXAGON_S2_cl0(0);
  // CHECK: @llvm.hexagon.S2.cl0p
  __builtin_HEXAGON_S2_cl0p(0);
  // CHECK: @llvm.hexagon.S2.cl1
  __builtin_HEXAGON_S2_cl1(0);
  // CHECK: @llvm.hexagon.S2.cl1p
  __builtin_HEXAGON_S2_cl1p(0);
  // CHECK: @llvm.hexagon.S2.clb
  __builtin_HEXAGON_S2_clb(0);
  // CHECK: @llvm.hexagon.S2.clbnorm
  __builtin_HEXAGON_S2_clbnorm(0);
  // CHECK: @llvm.hexagon.S2.clbp
  __builtin_HEXAGON_S2_clbp(0);
  // CHECK: @llvm.hexagon.S2.clrbit.i
  __builtin_HEXAGON_S2_clrbit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.clrbit.r
  __builtin_HEXAGON_S2_clrbit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.ct0
  __builtin_HEXAGON_S2_ct0(0);
  // CHECK: @llvm.hexagon.S2.ct0p
  __builtin_HEXAGON_S2_ct0p(0);
  // CHECK: @llvm.hexagon.S2.ct1
  __builtin_HEXAGON_S2_ct1(0);
  // CHECK: @llvm.hexagon.S2.ct1p
  __builtin_HEXAGON_S2_ct1p(0);
  // CHECK: @llvm.hexagon.S2.deinterleave
  __builtin_HEXAGON_S2_deinterleave(0);
  // CHECK: @llvm.hexagon.S2.extractu
  __builtin_HEXAGON_S2_extractu(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.extractu.rp
  __builtin_HEXAGON_S2_extractu_rp(0, 0);
  // CHECK: @llvm.hexagon.S2.extractup
  __builtin_HEXAGON_S2_extractup(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.extractup.rp
  __builtin_HEXAGON_S2_extractup_rp(0, 0);
  // CHECK: @llvm.hexagon.S2.insert
  __builtin_HEXAGON_S2_insert(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.insert.rp
  __builtin_HEXAGON_S2_insert_rp(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.insertp
  __builtin_HEXAGON_S2_insertp(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.insertp.rp
  __builtin_HEXAGON_S2_insertp_rp(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.interleave
  __builtin_HEXAGON_S2_interleave(0);
  // CHECK: @llvm.hexagon.S2.lfsp
  __builtin_HEXAGON_S2_lfsp(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p
  __builtin_HEXAGON_S2_lsl_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.acc
  __builtin_HEXAGON_S2_lsl_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.and
  __builtin_HEXAGON_S2_lsl_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.nac
  __builtin_HEXAGON_S2_lsl_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.or
  __builtin_HEXAGON_S2_lsl_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.xor
  __builtin_HEXAGON_S2_lsl_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r
  __builtin_HEXAGON_S2_lsl_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.acc
  __builtin_HEXAGON_S2_lsl_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.and
  __builtin_HEXAGON_S2_lsl_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.nac
  __builtin_HEXAGON_S2_lsl_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.or
  __builtin_HEXAGON_S2_lsl_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.vh
  __builtin_HEXAGON_S2_lsl_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.vw
  __builtin_HEXAGON_S2_lsl_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p
  __builtin_HEXAGON_S2_lsr_i_p(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.acc
  __builtin_HEXAGON_S2_lsr_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.and
  __builtin_HEXAGON_S2_lsr_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.nac
  __builtin_HEXAGON_S2_lsr_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.or
  __builtin_HEXAGON_S2_lsr_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.xacc
  __builtin_HEXAGON_S2_lsr_i_p_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r
  __builtin_HEXAGON_S2_lsr_i_r(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.acc
  __builtin_HEXAGON_S2_lsr_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.and
  __builtin_HEXAGON_S2_lsr_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.nac
  __builtin_HEXAGON_S2_lsr_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.or
  __builtin_HEXAGON_S2_lsr_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.xacc
  __builtin_HEXAGON_S2_lsr_i_r_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.vh
  __builtin_HEXAGON_S2_lsr_i_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.vw
  __builtin_HEXAGON_S2_lsr_i_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p
  __builtin_HEXAGON_S2_lsr_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.acc
  __builtin_HEXAGON_S2_lsr_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.and
  __builtin_HEXAGON_S2_lsr_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.nac
  __builtin_HEXAGON_S2_lsr_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.or
  __builtin_HEXAGON_S2_lsr_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.xor
  __builtin_HEXAGON_S2_lsr_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r
  __builtin_HEXAGON_S2_lsr_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.acc
  __builtin_HEXAGON_S2_lsr_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.and
  __builtin_HEXAGON_S2_lsr_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.nac
  __builtin_HEXAGON_S2_lsr_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.or
  __builtin_HEXAGON_S2_lsr_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.vh
  __builtin_HEXAGON_S2_lsr_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.vw
  __builtin_HEXAGON_S2_lsr_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.packhl
  __builtin_HEXAGON_S2_packhl(0, 0);
  // CHECK: @llvm.hexagon.S2.parityp
  __builtin_HEXAGON_S2_parityp(0, 0);
  // CHECK: @llvm.hexagon.S2.setbit.i
  __builtin_HEXAGON_S2_setbit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.setbit.r
  __builtin_HEXAGON_S2_setbit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffeb
  __builtin_HEXAGON_S2_shuffeb(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffeh
  __builtin_HEXAGON_S2_shuffeh(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffob
  __builtin_HEXAGON_S2_shuffob(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffoh
  __builtin_HEXAGON_S2_shuffoh(0, 0);
  // CHECK: @llvm.hexagon.S2.svsathb
  __builtin_HEXAGON_S2_svsathb(0);
  // CHECK: @llvm.hexagon.S2.svsathub
  __builtin_HEXAGON_S2_svsathub(0);
  // CHECK: @llvm.hexagon.S2.tableidxb.goodsyntax
  __builtin_HEXAGON_S2_tableidxb_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.tableidxd.goodsyntax
  __builtin_HEXAGON_S2_tableidxd_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.tableidxh.goodsyntax
  __builtin_HEXAGON_S2_tableidxh_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.tableidxw.goodsyntax
  __builtin_HEXAGON_S2_tableidxw_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.togglebit.i
  __builtin_HEXAGON_S2_togglebit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.togglebit.r
  __builtin_HEXAGON_S2_togglebit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.tstbit.i
  __builtin_HEXAGON_S2_tstbit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.tstbit.r
  __builtin_HEXAGON_S2_tstbit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.valignib
  __builtin_HEXAGON_S2_valignib(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.valignrb
  __builtin_HEXAGON_S2_valignrb(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.vcnegh
  __builtin_HEXAGON_S2_vcnegh(0, 0);
  // CHECK: @llvm.hexagon.S2.vcrotate
  __builtin_HEXAGON_S2_vcrotate(0, 0);
  // CHECK: @llvm.hexagon.S2.vrcnegh
  __builtin_HEXAGON_S2_vrcnegh(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.vrndpackwh
  __builtin_HEXAGON_S2_vrndpackwh(0);
  // CHECK: @llvm.hexagon.S2.vrndpackwhs
  __builtin_HEXAGON_S2_vrndpackwhs(0);
  // CHECK: @llvm.hexagon.S2.vsathb
  __builtin_HEXAGON_S2_vsathb(0);
  // CHECK: @llvm.hexagon.S2.vsathb.nopack
  __builtin_HEXAGON_S2_vsathb_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsathub
  __builtin_HEXAGON_S2_vsathub(0);
  // CHECK: @llvm.hexagon.S2.vsathub.nopack
  __builtin_HEXAGON_S2_vsathub_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsatwh
  __builtin_HEXAGON_S2_vsatwh(0);
  // CHECK: @llvm.hexagon.S2.vsatwh.nopack
  __builtin_HEXAGON_S2_vsatwh_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsatwuh
  __builtin_HEXAGON_S2_vsatwuh(0);
  // CHECK: @llvm.hexagon.S2.vsatwuh.nopack
  __builtin_HEXAGON_S2_vsatwuh_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsplatrb
  __builtin_HEXAGON_S2_vsplatrb(0);
  // CHECK: @llvm.hexagon.S2.vsplatrh
  __builtin_HEXAGON_S2_vsplatrh(0);
  // CHECK: @llvm.hexagon.S2.vspliceib
  __builtin_HEXAGON_S2_vspliceib(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.vsplicerb
  __builtin_HEXAGON_S2_vsplicerb(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.vsxtbh
  __builtin_HEXAGON_S2_vsxtbh(0);
  // CHECK: @llvm.hexagon.S2.vsxthw
  __builtin_HEXAGON_S2_vsxthw(0);
  // CHECK: @llvm.hexagon.S2.vtrunehb
  __builtin_HEXAGON_S2_vtrunehb(0);
  // CHECK: @llvm.hexagon.S2.vtrunewh
  __builtin_HEXAGON_S2_vtrunewh(0, 0);
  // CHECK: @llvm.hexagon.S2.vtrunohb
  __builtin_HEXAGON_S2_vtrunohb(0);
  // CHECK: @llvm.hexagon.S2.vtrunowh
  __builtin_HEXAGON_S2_vtrunowh(0, 0);
  // CHECK: @llvm.hexagon.S2.vzxtbh
  __builtin_HEXAGON_S2_vzxtbh(0);
  // CHECK: @llvm.hexagon.S2.vzxthw
  __builtin_HEXAGON_S2_vzxthw(0);
  // CHECK: @llvm.hexagon.S4.addaddi
  __builtin_HEXAGON_S4_addaddi(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.addi.asl.ri
  __builtin_HEXAGON_S4_addi_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.addi.lsr.ri
  __builtin_HEXAGON_S4_addi_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.andi.asl.ri
  __builtin_HEXAGON_S4_andi_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.andi.lsr.ri
  __builtin_HEXAGON_S4_andi_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.clbaddi
  __builtin_HEXAGON_S4_clbaddi(0, 0);
  // CHECK: @llvm.hexagon.S4.clbpaddi
  __builtin_HEXAGON_S4_clbpaddi(0, 0);
  // CHECK: @llvm.hexagon.S4.clbpnorm
  __builtin_HEXAGON_S4_clbpnorm(0);
  // CHECK: @llvm.hexagon.S4.extract
  __builtin_HEXAGON_S4_extract(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.extract.rp
  __builtin_HEXAGON_S4_extract_rp(0, 0);
  // CHECK: @llvm.hexagon.S4.extractp
  __builtin_HEXAGON_S4_extractp(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.extractp.rp
  __builtin_HEXAGON_S4_extractp_rp(0, 0);
  // CHECK: @llvm.hexagon.S4.lsli
  __builtin_HEXAGON_S4_lsli(0, 0);
  // CHECK: @llvm.hexagon.S4.ntstbit.i
  __builtin_HEXAGON_S4_ntstbit_i(0, 0);
  // CHECK: @llvm.hexagon.S4.ntstbit.r
  __builtin_HEXAGON_S4_ntstbit_r(0, 0);
  // CHECK: @llvm.hexagon.S4.or.andi
  __builtin_HEXAGON_S4_or_andi(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.or.andix
  __builtin_HEXAGON_S4_or_andix(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.or.ori
  __builtin_HEXAGON_S4_or_ori(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.ori.asl.ri
  __builtin_HEXAGON_S4_ori_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.ori.lsr.ri
  __builtin_HEXAGON_S4_ori_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.parity
  __builtin_HEXAGON_S4_parity(0, 0);
  // CHECK: @llvm.hexagon.S4.subaddi
  __builtin_HEXAGON_S4_subaddi(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.subi.asl.ri
  __builtin_HEXAGON_S4_subi_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.subi.lsr.ri
  __builtin_HEXAGON_S4_subi_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.vrcrotate
  __builtin_HEXAGON_S4_vrcrotate(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.vrcrotate.acc
  __builtin_HEXAGON_S4_vrcrotate_acc(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S4.vxaddsubh
  __builtin_HEXAGON_S4_vxaddsubh(0, 0);
  // CHECK: @llvm.hexagon.S4.vxaddsubhr
  __builtin_HEXAGON_S4_vxaddsubhr(0, 0);
  // CHECK: @llvm.hexagon.S4.vxaddsubw
  __builtin_HEXAGON_S4_vxaddsubw(0, 0);
  // CHECK: @llvm.hexagon.S4.vxsubaddh
  __builtin_HEXAGON_S4_vxsubaddh(0, 0);
  // CHECK: @llvm.hexagon.S4.vxsubaddhr
  __builtin_HEXAGON_S4_vxsubaddhr(0, 0);
  // CHECK: @llvm.hexagon.S4.vxsubaddw
  __builtin_HEXAGON_S4_vxsubaddw(0, 0);
  // CHECK: @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax
  __builtin_HEXAGON_S5_asrhub_rnd_sat_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S5.asrhub.sat
  __builtin_HEXAGON_S5_asrhub_sat(0, 0);
  // CHECK: @llvm.hexagon.S5.popcountp
  __builtin_HEXAGON_S5_popcountp(0);
  // CHECK: @llvm.hexagon.S5.vasrhrnd.goodsyntax
  __builtin_HEXAGON_S5_vasrhrnd_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p
  __builtin_HEXAGON_S6_rol_i_p(0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.acc
  __builtin_HEXAGON_S6_rol_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.and
  __builtin_HEXAGON_S6_rol_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.nac
  __builtin_HEXAGON_S6_rol_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.or
  __builtin_HEXAGON_S6_rol_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.xacc
  __builtin_HEXAGON_S6_rol_i_p_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r
  __builtin_HEXAGON_S6_rol_i_r(0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.acc
  __builtin_HEXAGON_S6_rol_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.and
  __builtin_HEXAGON_S6_rol_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.nac
  __builtin_HEXAGON_S6_rol_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.or
  __builtin_HEXAGON_S6_rol_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.xacc
  __builtin_HEXAGON_S6_rol_i_r_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.vsplatrbp
  __builtin_HEXAGON_S6_vsplatrbp(0);
  // CHECK: @llvm.hexagon.S6.vtrunehb.ppp
  __builtin_HEXAGON_S6_vtrunehb_ppp(0, 0);
  // CHECK: @llvm.hexagon.S6.vtrunohb.ppp
  __builtin_HEXAGON_S6_vtrunohb_ppp(0, 0);
  // CHECK: @llvm.hexagon.Y2.dccleana
  __builtin_HEXAGON_Y2_dccleana(0);
  // CHECK: @llvm.hexagon.Y2.dccleaninva
  __builtin_HEXAGON_Y2_dccleaninva(0);
  // CHECK: @llvm.hexagon.Y2.dcinva
  __builtin_HEXAGON_Y2_dcinva(0);
  // CHECK: @llvm.hexagon.Y2.dczeroa
  __builtin_HEXAGON_Y2_dczeroa(0);
  // CHECK: @llvm.hexagon.Y4.l2fetch
  __builtin_HEXAGON_Y4_l2fetch(0, 0);
  // CHECK: @llvm.hexagon.Y5.l2fetch
  __builtin_HEXAGON_Y5_l2fetch(0, 0);

  // CHECK: @llvm.hexagon.L2.loadrb.pbr
  __builtin_brev_ldb(0, 0, 0);
  // CHECK: @llvm.hexagon.L2.loadrd.pbr
  __builtin_brev_ldd(0, 0, 0);
  // CHECK: @llvm.hexagon.L2.loadrh.pbr
  __builtin_brev_ldh(0, 0, 0);
  // CHECK: @llvm.hexagon.L2.loadrub.pbr
  __builtin_brev_ldub(0, 0, 0);
  // CHECK: @llvm.hexagon.L2.loadruh.pbr
  __builtin_brev_lduh(0, 0, 0);
  // CHECK: @llvm.hexagon.L2.loadri.pbr
  __builtin_brev_ldw(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.storerb.pbr
  __builtin_brev_stb(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.storerd.pbr
  __builtin_brev_std(0, 0LL, 0);
  // CHECK: @llvm.hexagon.S2.storerh.pbr
  __builtin_brev_sth(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.storerf.pbr
  __builtin_brev_sthhi(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.storeri.pbr
  __builtin_brev_stw(0, 0, 0);

  // CHECK: @llvm.hexagon.circ.ldb
  __builtin_circ_ldb(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.ldd
  __builtin_circ_ldd(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.ldh
  __builtin_circ_ldh(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.ldub
  __builtin_circ_ldub(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.lduh
  __builtin_circ_lduh(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.ldw
  __builtin_circ_ldw(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.stb
  __builtin_circ_stb(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.std
  __builtin_circ_std(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.sth
  __builtin_circ_sth(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.sthhi
  __builtin_circ_sthhi(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.circ.stw
  __builtin_circ_stw(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.prefetch
  __builtin_HEXAGON_prefetch(0);
}
