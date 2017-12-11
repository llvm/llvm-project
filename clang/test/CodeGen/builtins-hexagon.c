// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon-unknown-elf -emit-llvm %s -o - | FileCheck %s

void foo() {
  int v16 __attribute__((__vector_size__(64)));
  int v32 __attribute__((__vector_size__(128)));
  int v64 __attribute__((__vector_size__(256)));

  // The circ/brev intrinsics do not have _HEXAGON_ in the name.
  __builtin_brev_ldb(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.ldb
  __builtin_brev_ldd(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.ldd
  __builtin_brev_ldh(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.ldh
  __builtin_brev_ldub(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.ldub
  __builtin_brev_lduh(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.lduh
  __builtin_brev_ldw(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.ldw
  __builtin_brev_stb(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.stb
  __builtin_brev_std(0, 0LL, 0);
  // CHECK: @llvm.hexagon.brev.std
  __builtin_brev_sth(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.sth
  __builtin_brev_sthhi(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.sthhi
  __builtin_brev_stw(0, 0, 0);
  // CHECK: @llvm.hexagon.brev.stw
  __builtin_circ_ldb(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.ldb
  __builtin_circ_ldd(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.ldd
  __builtin_circ_ldh(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.ldh
  __builtin_circ_ldub(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.ldub
  __builtin_circ_lduh(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.lduh
  __builtin_circ_ldw(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.ldw
  __builtin_circ_stb(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.stb
  __builtin_circ_std(0, 0LL, 0, 0);
  // CHECK: llvm.hexagon.circ.std
  __builtin_circ_sth(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.sth
  __builtin_circ_sthhi(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.sthhi
  __builtin_circ_stw(0, 0, 0, 0);
  // CHECK: llvm.hexagon.circ.stw

  __builtin_HEXAGON_A2_abs(0);
  // CHECK: @llvm.hexagon.A2.abs
  __builtin_HEXAGON_A2_absp(0);
  // CHECK: @llvm.hexagon.A2.absp
  __builtin_HEXAGON_A2_abssat(0);
  // CHECK: @llvm.hexagon.A2.abssat
  __builtin_HEXAGON_A2_add(0, 0);
  // CHECK: @llvm.hexagon.A2.add
  __builtin_HEXAGON_A2_addh_h16_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.hh
  __builtin_HEXAGON_A2_addh_h16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.hl
  __builtin_HEXAGON_A2_addh_h16_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.lh
  __builtin_HEXAGON_A2_addh_h16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.ll
  __builtin_HEXAGON_A2_addh_h16_sat_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.hh
  __builtin_HEXAGON_A2_addh_h16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.hl
  __builtin_HEXAGON_A2_addh_h16_sat_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.lh
  __builtin_HEXAGON_A2_addh_h16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.h16.sat.ll
  __builtin_HEXAGON_A2_addh_l16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.hl
  __builtin_HEXAGON_A2_addh_l16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.ll
  __builtin_HEXAGON_A2_addh_l16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.sat.hl
  __builtin_HEXAGON_A2_addh_l16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.addh.l16.sat.ll
  __builtin_HEXAGON_A2_addi(0, 0);
  // CHECK: @llvm.hexagon.A2.addi
  __builtin_HEXAGON_A2_addp(0, 0);
  // CHECK: @llvm.hexagon.A2.addp
  __builtin_HEXAGON_A2_addpsat(0, 0);
  // CHECK: @llvm.hexagon.A2.addpsat
  __builtin_HEXAGON_A2_addsat(0, 0);
  // CHECK: @llvm.hexagon.A2.addsat
  __builtin_HEXAGON_A2_addsp(0, 0);
  // CHECK: @llvm.hexagon.A2.addsp
  __builtin_HEXAGON_A2_and(0, 0);
  // CHECK: @llvm.hexagon.A2.and
  __builtin_HEXAGON_A2_andir(0, 0);
  // CHECK: @llvm.hexagon.A2.andir
  __builtin_HEXAGON_A2_andp(0, 0);
  // CHECK: @llvm.hexagon.A2.andp
  __builtin_HEXAGON_A2_aslh(0);
  // CHECK: @llvm.hexagon.A2.aslh
  __builtin_HEXAGON_A2_asrh(0);
  // CHECK: @llvm.hexagon.A2.asrh
  __builtin_HEXAGON_A2_combine_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.combine.hh
  __builtin_HEXAGON_A2_combine_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.combine.hl
  __builtin_HEXAGON_A2_combineii(0, 0);
  // CHECK: @llvm.hexagon.A2.combineii
  __builtin_HEXAGON_A2_combine_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.combine.lh
  __builtin_HEXAGON_A2_combine_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.combine.ll
  __builtin_HEXAGON_A2_combinew(0, 0);
  // CHECK: @llvm.hexagon.A2.combinew
  __builtin_HEXAGON_A2_max(0, 0);
  // CHECK: @llvm.hexagon.A2.max
  __builtin_HEXAGON_A2_maxp(0, 0);
  // CHECK: @llvm.hexagon.A2.maxp
  __builtin_HEXAGON_A2_maxu(0, 0);
  // CHECK: @llvm.hexagon.A2.maxu
  __builtin_HEXAGON_A2_maxup(0, 0);
  // CHECK: @llvm.hexagon.A2.maxup
  __builtin_HEXAGON_A2_min(0, 0);
  // CHECK: @llvm.hexagon.A2.min
  __builtin_HEXAGON_A2_minp(0, 0);
  // CHECK: @llvm.hexagon.A2.minp
  __builtin_HEXAGON_A2_minu(0, 0);
  // CHECK: @llvm.hexagon.A2.minu
  __builtin_HEXAGON_A2_minup(0, 0);
  // CHECK: @llvm.hexagon.A2.minup
  __builtin_HEXAGON_A2_neg(0);
  // CHECK: @llvm.hexagon.A2.neg
  __builtin_HEXAGON_A2_negp(0);
  // CHECK: @llvm.hexagon.A2.negp
  __builtin_HEXAGON_A2_negsat(0);
  // CHECK: @llvm.hexagon.A2.negsat
  __builtin_HEXAGON_A2_not(0);
  // CHECK: @llvm.hexagon.A2.not
  __builtin_HEXAGON_A2_notp(0);
  // CHECK: @llvm.hexagon.A2.notp
  __builtin_HEXAGON_A2_or(0, 0);
  // CHECK: @llvm.hexagon.A2.or
  __builtin_HEXAGON_A2_orir(0, 0);
  // CHECK: @llvm.hexagon.A2.orir
  __builtin_HEXAGON_A2_orp(0, 0);
  // CHECK: @llvm.hexagon.A2.orp
  __builtin_HEXAGON_A2_roundsat(0);
  // CHECK: @llvm.hexagon.A2.roundsat
  __builtin_HEXAGON_A2_sat(0);
  // CHECK: @llvm.hexagon.A2.sat
  __builtin_HEXAGON_A2_satb(0);
  // CHECK: @llvm.hexagon.A2.satb
  __builtin_HEXAGON_A2_sath(0);
  // CHECK: @llvm.hexagon.A2.sath
  __builtin_HEXAGON_A2_satub(0);
  // CHECK: @llvm.hexagon.A2.satub
  __builtin_HEXAGON_A2_satuh(0);
  // CHECK: @llvm.hexagon.A2.satuh
  __builtin_HEXAGON_A2_sub(0, 0);
  // CHECK: @llvm.hexagon.A2.sub
  __builtin_HEXAGON_A2_subh_h16_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.hh
  __builtin_HEXAGON_A2_subh_h16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.hl
  __builtin_HEXAGON_A2_subh_h16_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.lh
  __builtin_HEXAGON_A2_subh_h16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.ll
  __builtin_HEXAGON_A2_subh_h16_sat_hh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.hh
  __builtin_HEXAGON_A2_subh_h16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.hl
  __builtin_HEXAGON_A2_subh_h16_sat_lh(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.lh
  __builtin_HEXAGON_A2_subh_h16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.h16.sat.ll
  __builtin_HEXAGON_A2_subh_l16_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.hl
  __builtin_HEXAGON_A2_subh_l16_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.ll
  __builtin_HEXAGON_A2_subh_l16_sat_hl(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.sat.hl
  __builtin_HEXAGON_A2_subh_l16_sat_ll(0, 0);
  // CHECK: @llvm.hexagon.A2.subh.l16.sat.ll
  __builtin_HEXAGON_A2_subp(0, 0);
  // CHECK: @llvm.hexagon.A2.subp
  __builtin_HEXAGON_A2_subri(0, 0);
  // CHECK: @llvm.hexagon.A2.subri
  __builtin_HEXAGON_A2_subsat(0, 0);
  // CHECK: @llvm.hexagon.A2.subsat
  __builtin_HEXAGON_A2_svaddh(0, 0);
  // CHECK: @llvm.hexagon.A2.svaddh
  __builtin_HEXAGON_A2_svaddhs(0, 0);
  // CHECK: @llvm.hexagon.A2.svaddhs
  __builtin_HEXAGON_A2_svadduhs(0, 0);
  // CHECK: @llvm.hexagon.A2.svadduhs
  __builtin_HEXAGON_A2_svavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.svavgh
  __builtin_HEXAGON_A2_svavghs(0, 0);
  // CHECK: @llvm.hexagon.A2.svavghs
  __builtin_HEXAGON_A2_svnavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.svnavgh
  __builtin_HEXAGON_A2_svsubh(0, 0);
  // CHECK: @llvm.hexagon.A2.svsubh
  __builtin_HEXAGON_A2_svsubhs(0, 0);
  // CHECK: @llvm.hexagon.A2.svsubhs
  __builtin_HEXAGON_A2_svsubuhs(0, 0);
  // CHECK: @llvm.hexagon.A2.svsubuhs
  __builtin_HEXAGON_A2_swiz(0);
  // CHECK: @llvm.hexagon.A2.swiz
  __builtin_HEXAGON_A2_sxtb(0);
  // CHECK: @llvm.hexagon.A2.sxtb
  __builtin_HEXAGON_A2_sxth(0);
  // CHECK: @llvm.hexagon.A2.sxth
  __builtin_HEXAGON_A2_sxtw(0);
  // CHECK: @llvm.hexagon.A2.sxtw
  __builtin_HEXAGON_A2_tfr(0);
  // CHECK: @llvm.hexagon.A2.tfr
  __builtin_HEXAGON_A2_tfrih(0, 0);
  // CHECK: @llvm.hexagon.A2.tfrih
  __builtin_HEXAGON_A2_tfril(0, 0);
  // CHECK: @llvm.hexagon.A2.tfril
  __builtin_HEXAGON_A2_tfrp(0);
  // CHECK: @llvm.hexagon.A2.tfrp
  __builtin_HEXAGON_A2_tfrpi(0);
  // CHECK: @llvm.hexagon.A2.tfrpi
  __builtin_HEXAGON_A2_tfrsi(0);
  // CHECK: @llvm.hexagon.A2.tfrsi
  __builtin_HEXAGON_A2_vabsh(0);
  // CHECK: @llvm.hexagon.A2.vabsh
  __builtin_HEXAGON_A2_vabshsat(0);
  // CHECK: @llvm.hexagon.A2.vabshsat
  __builtin_HEXAGON_A2_vabsw(0);
  // CHECK: @llvm.hexagon.A2.vabsw
  __builtin_HEXAGON_A2_vabswsat(0);
  // CHECK: @llvm.hexagon.A2.vabswsat
  __builtin_HEXAGON_A2_vaddb_map(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddb.map
  __builtin_HEXAGON_A2_vaddh(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddh
  __builtin_HEXAGON_A2_vaddhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddhs
  __builtin_HEXAGON_A2_vaddub(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddub
  __builtin_HEXAGON_A2_vaddubs(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddubs
  __builtin_HEXAGON_A2_vadduhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vadduhs
  __builtin_HEXAGON_A2_vaddw(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddw
  __builtin_HEXAGON_A2_vaddws(0, 0);
  // CHECK: @llvm.hexagon.A2.vaddws
  __builtin_HEXAGON_A2_vavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgh
  __builtin_HEXAGON_A2_vavghcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavghcr
  __builtin_HEXAGON_A2_vavghr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavghr
  __builtin_HEXAGON_A2_vavgub(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgub
  __builtin_HEXAGON_A2_vavgubr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgubr
  __builtin_HEXAGON_A2_vavguh(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguh
  __builtin_HEXAGON_A2_vavguhr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguhr
  __builtin_HEXAGON_A2_vavguw(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguw
  __builtin_HEXAGON_A2_vavguwr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavguwr
  __builtin_HEXAGON_A2_vavgw(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgw
  __builtin_HEXAGON_A2_vavgwcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgwcr
  __builtin_HEXAGON_A2_vavgwr(0, 0);
  // CHECK: @llvm.hexagon.A2.vavgwr
  __builtin_HEXAGON_A2_vcmpbeq(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpbeq
  __builtin_HEXAGON_A2_vcmpbgtu(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpbgtu
  __builtin_HEXAGON_A2_vcmpheq(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpheq
  __builtin_HEXAGON_A2_vcmphgt(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmphgt
  __builtin_HEXAGON_A2_vcmphgtu(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmphgtu
  __builtin_HEXAGON_A2_vcmpweq(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpweq
  __builtin_HEXAGON_A2_vcmpwgt(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpwgt
  __builtin_HEXAGON_A2_vcmpwgtu(0, 0);
  // CHECK: @llvm.hexagon.A2.vcmpwgtu
  __builtin_HEXAGON_A2_vconj(0);
  // CHECK: @llvm.hexagon.A2.vconj
  __builtin_HEXAGON_A2_vmaxb(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxb
  __builtin_HEXAGON_A2_vmaxh(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxh
  __builtin_HEXAGON_A2_vmaxub(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxub
  __builtin_HEXAGON_A2_vmaxuh(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxuh
  __builtin_HEXAGON_A2_vmaxuw(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxuw
  __builtin_HEXAGON_A2_vmaxw(0, 0);
  // CHECK: @llvm.hexagon.A2.vmaxw
  __builtin_HEXAGON_A2_vminb(0, 0);
  // CHECK: @llvm.hexagon.A2.vminb
  __builtin_HEXAGON_A2_vminh(0, 0);
  // CHECK: @llvm.hexagon.A2.vminh
  __builtin_HEXAGON_A2_vminub(0, 0);
  // CHECK: @llvm.hexagon.A2.vminub
  __builtin_HEXAGON_A2_vminuh(0, 0);
  // CHECK: @llvm.hexagon.A2.vminuh
  __builtin_HEXAGON_A2_vminuw(0, 0);
  // CHECK: @llvm.hexagon.A2.vminuw
  __builtin_HEXAGON_A2_vminw(0, 0);
  // CHECK: @llvm.hexagon.A2.vminw
  __builtin_HEXAGON_A2_vnavgh(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgh
  __builtin_HEXAGON_A2_vnavghcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavghcr
  __builtin_HEXAGON_A2_vnavghr(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavghr
  __builtin_HEXAGON_A2_vnavgw(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgw
  __builtin_HEXAGON_A2_vnavgwcr(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgwcr
  __builtin_HEXAGON_A2_vnavgwr(0, 0);
  // CHECK: @llvm.hexagon.A2.vnavgwr
  __builtin_HEXAGON_A2_vraddub(0, 0);
  // CHECK: @llvm.hexagon.A2.vraddub
  __builtin_HEXAGON_A2_vraddub_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.A2.vraddub.acc
  __builtin_HEXAGON_A2_vrsadub(0, 0);
  // CHECK: @llvm.hexagon.A2.vrsadub
  __builtin_HEXAGON_A2_vrsadub_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.A2.vrsadub.acc
  __builtin_HEXAGON_A2_vsubb_map(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubb.map
  __builtin_HEXAGON_A2_vsubh(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubh
  __builtin_HEXAGON_A2_vsubhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubhs
  __builtin_HEXAGON_A2_vsubub(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubub
  __builtin_HEXAGON_A2_vsububs(0, 0);
  // CHECK: @llvm.hexagon.A2.vsububs
  __builtin_HEXAGON_A2_vsubuhs(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubuhs
  __builtin_HEXAGON_A2_vsubw(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubw
  __builtin_HEXAGON_A2_vsubws(0, 0);
  // CHECK: @llvm.hexagon.A2.vsubws
  __builtin_HEXAGON_A2_xor(0, 0);
  // CHECK: @llvm.hexagon.A2.xor
  __builtin_HEXAGON_A2_xorp(0, 0);
  // CHECK: @llvm.hexagon.A2.xorp
  __builtin_HEXAGON_A2_zxtb(0);
  // CHECK: @llvm.hexagon.A2.zxtb
  __builtin_HEXAGON_A2_zxth(0);
  // CHECK: @llvm.hexagon.A2.zxth
  __builtin_HEXAGON_A4_andn(0, 0);
  // CHECK: @llvm.hexagon.A4.andn
  __builtin_HEXAGON_A4_andnp(0, 0);
  // CHECK: @llvm.hexagon.A4.andnp
  __builtin_HEXAGON_A4_bitsplit(0, 0);
  // CHECK: @llvm.hexagon.A4.bitsplit
  __builtin_HEXAGON_A4_bitspliti(0, 0);
  // CHECK: @llvm.hexagon.A4.bitspliti
  __builtin_HEXAGON_A4_boundscheck(0, 0);
  // CHECK: @llvm.hexagon.A4.boundscheck
  __builtin_HEXAGON_A4_cmpbeq(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbeq
  __builtin_HEXAGON_A4_cmpbeqi(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbeqi
  __builtin_HEXAGON_A4_cmpbgt(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgt
  __builtin_HEXAGON_A4_cmpbgti(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgti
  __builtin_HEXAGON_A4_cmpbgtu(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgtu
  __builtin_HEXAGON_A4_cmpbgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpbgtui
  __builtin_HEXAGON_A4_cmpheq(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpheq
  __builtin_HEXAGON_A4_cmpheqi(0, 0);
  // CHECK: @llvm.hexagon.A4.cmpheqi
  __builtin_HEXAGON_A4_cmphgt(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgt
  __builtin_HEXAGON_A4_cmphgti(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgti
  __builtin_HEXAGON_A4_cmphgtu(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgtu
  __builtin_HEXAGON_A4_cmphgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.cmphgtui
  __builtin_HEXAGON_A4_combineir(0, 0);
  // CHECK: @llvm.hexagon.A4.combineir
  __builtin_HEXAGON_A4_combineri(0, 0);
  // CHECK: @llvm.hexagon.A4.combineri
  __builtin_HEXAGON_A4_cround_ri(0, 0);
  // CHECK: @llvm.hexagon.A4.cround.ri
  __builtin_HEXAGON_A4_cround_rr(0, 0);
  // CHECK: @llvm.hexagon.A4.cround.rr
  __builtin_HEXAGON_A4_modwrapu(0, 0);
  // CHECK: @llvm.hexagon.A4.modwrapu
  __builtin_HEXAGON_A4_orn(0, 0);
  // CHECK: @llvm.hexagon.A4.orn
  __builtin_HEXAGON_A4_ornp(0, 0);
  // CHECK: @llvm.hexagon.A4.ornp
  __builtin_HEXAGON_A4_rcmpeq(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpeq
  __builtin_HEXAGON_A4_rcmpeqi(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpeqi
  __builtin_HEXAGON_A4_rcmpneq(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpneq
  __builtin_HEXAGON_A4_rcmpneqi(0, 0);
  // CHECK: @llvm.hexagon.A4.rcmpneqi
  __builtin_HEXAGON_A4_round_ri(0, 0);
  // CHECK: @llvm.hexagon.A4.round.ri
  __builtin_HEXAGON_A4_round_ri_sat(0, 0);
  // CHECK: @llvm.hexagon.A4.round.ri.sat
  __builtin_HEXAGON_A4_round_rr(0, 0);
  // CHECK: @llvm.hexagon.A4.round.rr
  __builtin_HEXAGON_A4_round_rr_sat(0, 0);
  // CHECK: @llvm.hexagon.A4.round.rr.sat
  __builtin_HEXAGON_A4_tlbmatch(0, 0);
  // CHECK: @llvm.hexagon.A4.tlbmatch
  __builtin_HEXAGON_A4_vcmpbeq_any(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbeq.any
  __builtin_HEXAGON_A4_vcmpbeqi(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbeqi
  __builtin_HEXAGON_A4_vcmpbgt(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbgt
  __builtin_HEXAGON_A4_vcmpbgti(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbgti
  __builtin_HEXAGON_A4_vcmpbgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpbgtui
  __builtin_HEXAGON_A4_vcmpheqi(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpheqi
  __builtin_HEXAGON_A4_vcmphgti(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmphgti
  __builtin_HEXAGON_A4_vcmphgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmphgtui
  __builtin_HEXAGON_A4_vcmpweqi(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpweqi
  __builtin_HEXAGON_A4_vcmpwgti(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpwgti
  __builtin_HEXAGON_A4_vcmpwgtui(0, 0);
  // CHECK: @llvm.hexagon.A4.vcmpwgtui
  __builtin_HEXAGON_A4_vrmaxh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxh
  __builtin_HEXAGON_A4_vrmaxuh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxuh
  __builtin_HEXAGON_A4_vrmaxuw(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxuw
  __builtin_HEXAGON_A4_vrmaxw(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrmaxw
  __builtin_HEXAGON_A4_vrminh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminh
  __builtin_HEXAGON_A4_vrminuh(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminuh
  __builtin_HEXAGON_A4_vrminuw(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminuw
  __builtin_HEXAGON_A4_vrminw(0, 0, 0);
  // CHECK: @llvm.hexagon.A4.vrminw
  __builtin_HEXAGON_A5_vaddhubs(0, 0);
  // CHECK: @llvm.hexagon.A5.vaddhubs
  __builtin_HEXAGON_C2_all8(0);
  // CHECK: @llvm.hexagon.C2.all8
  __builtin_HEXAGON_C2_and(0, 0);
  // CHECK: @llvm.hexagon.C2.and
  __builtin_HEXAGON_C2_andn(0, 0);
  // CHECK: @llvm.hexagon.C2.andn
  __builtin_HEXAGON_C2_any8(0);
  // CHECK: @llvm.hexagon.C2.any8
  __builtin_HEXAGON_C2_bitsclr(0, 0);
  // CHECK: @llvm.hexagon.C2.bitsclr
  __builtin_HEXAGON_C2_bitsclri(0, 0);
  // CHECK: @llvm.hexagon.C2.bitsclri
  __builtin_HEXAGON_C2_bitsset(0, 0);
  // CHECK: @llvm.hexagon.C2.bitsset
  __builtin_HEXAGON_C2_cmpeq(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpeq
  __builtin_HEXAGON_C2_cmpeqi(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpeqi
  __builtin_HEXAGON_C2_cmpeqp(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpeqp
  __builtin_HEXAGON_C2_cmpgei(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgei
  __builtin_HEXAGON_C2_cmpgeui(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgeui
  __builtin_HEXAGON_C2_cmpgt(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgt
  __builtin_HEXAGON_C2_cmpgti(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgti
  __builtin_HEXAGON_C2_cmpgtp(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtp
  __builtin_HEXAGON_C2_cmpgtu(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtu
  __builtin_HEXAGON_C2_cmpgtui(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtui
  __builtin_HEXAGON_C2_cmpgtup(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpgtup
  __builtin_HEXAGON_C2_cmplt(0, 0);
  // CHECK: @llvm.hexagon.C2.cmplt
  __builtin_HEXAGON_C2_cmpltu(0, 0);
  // CHECK: @llvm.hexagon.C2.cmpltu
  __builtin_HEXAGON_C2_mask(0);
  // CHECK: @llvm.hexagon.C2.mask
  __builtin_HEXAGON_C2_mux(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.mux
  __builtin_HEXAGON_C2_muxii(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.muxii
  __builtin_HEXAGON_C2_muxir(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.muxir
  __builtin_HEXAGON_C2_muxri(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.muxri
  __builtin_HEXAGON_C2_not(0);
  // CHECK: @llvm.hexagon.C2.not
  __builtin_HEXAGON_C2_or (0, 0);
  // CHECK: @llvm.hexagon.C2.or 
  __builtin_HEXAGON_C2_orn(0, 0);
  // CHECK: @llvm.hexagon.C2.orn
  __builtin_HEXAGON_C2_pxfer_map(0);
  // CHECK: @llvm.hexagon.C2.pxfer.map
  __builtin_HEXAGON_C2_tfrpr(0);
  // CHECK: @llvm.hexagon.C2.tfrpr
  __builtin_HEXAGON_C2_tfrrp(0);
  // CHECK: @llvm.hexagon.C2.tfrrp
  __builtin_HEXAGON_C2_vitpack(0, 0);
  // CHECK: @llvm.hexagon.C2.vitpack
  __builtin_HEXAGON_C2_vmux(0, 0, 0);
  // CHECK: @llvm.hexagon.C2.vmux
  __builtin_HEXAGON_C2_xor(0, 0);
  // CHECK: @llvm.hexagon.C2.xor
  __builtin_HEXAGON_C4_and_and(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.and.and
  __builtin_HEXAGON_C4_and_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.and.andn
  __builtin_HEXAGON_C4_and_or(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.and.or
  __builtin_HEXAGON_C4_and_orn(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.and.orn
  __builtin_HEXAGON_C4_cmplte(0, 0);
  // CHECK: @llvm.hexagon.C4.cmplte
  __builtin_HEXAGON_C4_cmpltei(0, 0);
  // CHECK: @llvm.hexagon.C4.cmpltei
  __builtin_HEXAGON_C4_cmplteu(0, 0);
  // CHECK: @llvm.hexagon.C4.cmplteu
  __builtin_HEXAGON_C4_cmplteui(0, 0);
  // CHECK: @llvm.hexagon.C4.cmplteui
  __builtin_HEXAGON_C4_cmpneq(0, 0);
  // CHECK: @llvm.hexagon.C4.cmpneq
  __builtin_HEXAGON_C4_cmpneqi(0, 0);
  // CHECK: @llvm.hexagon.C4.cmpneqi
  __builtin_HEXAGON_C4_fastcorner9(0, 0);
  // CHECK: @llvm.hexagon.C4.fastcorner9
  __builtin_HEXAGON_C4_fastcorner9_not(0, 0);
  // CHECK: @llvm.hexagon.C4.fastcorner9.not
  __builtin_HEXAGON_C4_nbitsclr(0, 0);
  // CHECK: @llvm.hexagon.C4.nbitsclr
  __builtin_HEXAGON_C4_nbitsclri(0, 0);
  // CHECK: @llvm.hexagon.C4.nbitsclri
  __builtin_HEXAGON_C4_nbitsset(0, 0);
  // CHECK: @llvm.hexagon.C4.nbitsset
  __builtin_HEXAGON_C4_or_and(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.or.and
  __builtin_HEXAGON_C4_or_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.or.andn
  __builtin_HEXAGON_C4_or_or(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.or.or
  __builtin_HEXAGON_C4_or_orn(0, 0, 0);
  // CHECK: @llvm.hexagon.C4.or.orn
  __builtin_HEXAGON_F2_conv_d2df(0);
  // CHECK: @llvm.hexagon.F2.conv.d2df
  __builtin_HEXAGON_F2_conv_d2sf(0);
  // CHECK: @llvm.hexagon.F2.conv.d2sf
  __builtin_HEXAGON_F2_conv_df2d(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2d
  __builtin_HEXAGON_F2_conv_df2d_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2d.chop
  __builtin_HEXAGON_F2_conv_df2sf(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2sf
  __builtin_HEXAGON_F2_conv_df2ud(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2ud
  __builtin_HEXAGON_F2_conv_df2ud_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2ud.chop
  __builtin_HEXAGON_F2_conv_df2uw(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2uw
  __builtin_HEXAGON_F2_conv_df2uw_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2uw.chop
  __builtin_HEXAGON_F2_conv_df2w(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2w
  __builtin_HEXAGON_F2_conv_df2w_chop(0.0);
  // CHECK: @llvm.hexagon.F2.conv.df2w.chop
  __builtin_HEXAGON_F2_conv_sf2d(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2d
  __builtin_HEXAGON_F2_conv_sf2d_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2d.chop
  __builtin_HEXAGON_F2_conv_sf2df(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2df
  __builtin_HEXAGON_F2_conv_sf2ud(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2ud
  __builtin_HEXAGON_F2_conv_sf2ud_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2ud.chop
  __builtin_HEXAGON_F2_conv_sf2uw(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2uw
  __builtin_HEXAGON_F2_conv_sf2uw_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2uw.chop
  __builtin_HEXAGON_F2_conv_sf2w(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2w
  __builtin_HEXAGON_F2_conv_sf2w_chop(0.0f);
  // CHECK: @llvm.hexagon.F2.conv.sf2w.chop
  __builtin_HEXAGON_F2_conv_ud2df(0);
  // CHECK: @llvm.hexagon.F2.conv.ud2df
  __builtin_HEXAGON_F2_conv_ud2sf(0);
  // CHECK: @llvm.hexagon.F2.conv.ud2sf
  __builtin_HEXAGON_F2_conv_uw2df(0);
  // CHECK: @llvm.hexagon.F2.conv.uw2df
  __builtin_HEXAGON_F2_conv_uw2sf(0);
  // CHECK: @llvm.hexagon.F2.conv.uw2sf
  __builtin_HEXAGON_F2_conv_w2df(0);
  // CHECK: @llvm.hexagon.F2.conv.w2df
  __builtin_HEXAGON_F2_conv_w2sf(0);
  // CHECK: @llvm.hexagon.F2.conv.w2sf
  __builtin_HEXAGON_F2_dfclass(0.0, 0);
  // CHECK: @llvm.hexagon.F2.dfclass
  __builtin_HEXAGON_F2_dfcmpeq(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfcmpeq
  __builtin_HEXAGON_F2_dfcmpge(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfcmpge
  __builtin_HEXAGON_F2_dfcmpgt(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfcmpgt
  __builtin_HEXAGON_F2_dfcmpuo(0.0, 0.0);
  // CHECK: @llvm.hexagon.F2.dfcmpuo
  __builtin_HEXAGON_F2_dfimm_n(0);
  // CHECK: @llvm.hexagon.F2.dfimm.n
  __builtin_HEXAGON_F2_dfimm_p(0);
  // CHECK: @llvm.hexagon.F2.dfimm.p
  __builtin_HEXAGON_F2_sfadd(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfadd
  __builtin_HEXAGON_F2_sfclass(0.0f, 0);
  // CHECK: @llvm.hexagon.F2.sfclass
  __builtin_HEXAGON_F2_sfcmpeq(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfcmpeq
  __builtin_HEXAGON_F2_sfcmpge(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfcmpge
  __builtin_HEXAGON_F2_sfcmpgt(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfcmpgt
  __builtin_HEXAGON_F2_sfcmpuo(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfcmpuo
  __builtin_HEXAGON_F2_sffixupd(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffixupd
  __builtin_HEXAGON_F2_sffixupn(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffixupn
  __builtin_HEXAGON_F2_sffixupr(0.0f);
  // CHECK: @llvm.hexagon.F2.sffixupr
  __builtin_HEXAGON_F2_sffma(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffma
  __builtin_HEXAGON_F2_sffma_lib(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffma.lib
  __builtin_HEXAGON_F2_sffma_sc(0.0f, 0.0f, 0.0f, 0);
  // CHECK: @llvm.hexagon.F2.sffma.sc
  __builtin_HEXAGON_F2_sffms(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffms
  __builtin_HEXAGON_F2_sffms_lib(0.0f, 0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sffms.lib
  __builtin_HEXAGON_F2_sfimm_n(0);
  // CHECK: @llvm.hexagon.F2.sfimm.n
  __builtin_HEXAGON_F2_sfimm_p(0);
  // CHECK: @llvm.hexagon.F2.sfimm.p
  __builtin_HEXAGON_F2_sfmax(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfmax
  __builtin_HEXAGON_F2_sfmin(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfmin
  __builtin_HEXAGON_F2_sfmpy(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfmpy
  __builtin_HEXAGON_F2_sfsub(0.0f, 0.0f);
  // CHECK: @llvm.hexagon.F2.sfsub
  __builtin_HEXAGON_M2_acci(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.acci
  __builtin_HEXAGON_M2_accii(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.accii
  __builtin_HEXAGON_M2_cmaci_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmaci.s0
  __builtin_HEXAGON_M2_cmacr_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacr.s0
  __builtin_HEXAGON_M2_cmacsc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacsc.s0
  __builtin_HEXAGON_M2_cmacsc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacsc.s1
  __builtin_HEXAGON_M2_cmacs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacs.s0
  __builtin_HEXAGON_M2_cmacs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cmacs.s1
  __builtin_HEXAGON_M2_cmpyi_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyi.s0
  __builtin_HEXAGON_M2_cmpyr_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyr.s0
  __builtin_HEXAGON_M2_cmpyrsc_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrsc.s0
  __builtin_HEXAGON_M2_cmpyrsc_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrsc.s1
  __builtin_HEXAGON_M2_cmpyrs_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrs.s0
  __builtin_HEXAGON_M2_cmpyrs_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpyrs.s1
  __builtin_HEXAGON_M2_cmpysc_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpysc.s0
  __builtin_HEXAGON_M2_cmpysc_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpysc.s1
  __builtin_HEXAGON_M2_cmpys_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpys.s0
  __builtin_HEXAGON_M2_cmpys_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.cmpys.s1
  __builtin_HEXAGON_M2_cnacsc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cnacsc.s0
  __builtin_HEXAGON_M2_cnacsc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cnacsc.s1
  __builtin_HEXAGON_M2_cnacs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cnacs.s0
  __builtin_HEXAGON_M2_cnacs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.cnacs.s1
  __builtin_HEXAGON_M2_dpmpyss_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.acc.s0
  __builtin_HEXAGON_M2_dpmpyss_nac_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.nac.s0
  __builtin_HEXAGON_M2_dpmpyss_rnd_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.rnd.s0
  __builtin_HEXAGON_M2_dpmpyss_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyss.s0
  __builtin_HEXAGON_M2_dpmpyuu_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyuu.acc.s0
  __builtin_HEXAGON_M2_dpmpyuu_nac_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyuu.nac.s0
  __builtin_HEXAGON_M2_dpmpyuu_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.dpmpyuu.s0
  __builtin_HEXAGON_M2_hmmpyh_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyh.rs1
  __builtin_HEXAGON_M2_hmmpyh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyh.s1
  __builtin_HEXAGON_M2_hmmpyl_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyl.rs1
  __builtin_HEXAGON_M2_hmmpyl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.hmmpyl.s1
  __builtin_HEXAGON_M2_maci(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.maci
  __builtin_HEXAGON_M2_macsin(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.macsin
  __builtin_HEXAGON_M2_macsip(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.macsip
  __builtin_HEXAGON_M2_mmachs_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.rs0
  __builtin_HEXAGON_M2_mmachs_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.rs1
  __builtin_HEXAGON_M2_mmachs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.s0
  __builtin_HEXAGON_M2_mmachs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmachs.s1
  __builtin_HEXAGON_M2_mmacls_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.rs0
  __builtin_HEXAGON_M2_mmacls_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.rs1
  __builtin_HEXAGON_M2_mmacls_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.s0
  __builtin_HEXAGON_M2_mmacls_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacls.s1
  __builtin_HEXAGON_M2_mmacuhs_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.rs0
  __builtin_HEXAGON_M2_mmacuhs_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.rs1
  __builtin_HEXAGON_M2_mmacuhs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.s0
  __builtin_HEXAGON_M2_mmacuhs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmacuhs.s1
  __builtin_HEXAGON_M2_mmaculs_rs0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.rs0
  __builtin_HEXAGON_M2_mmaculs_rs1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.rs1
  __builtin_HEXAGON_M2_mmaculs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.s0
  __builtin_HEXAGON_M2_mmaculs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mmaculs.s1
  __builtin_HEXAGON_M2_mmpyh_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.rs0
  __builtin_HEXAGON_M2_mmpyh_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.rs1
  __builtin_HEXAGON_M2_mmpyh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.s0
  __builtin_HEXAGON_M2_mmpyh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyh.s1
  __builtin_HEXAGON_M2_mmpyl_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.rs0
  __builtin_HEXAGON_M2_mmpyl_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.rs1
  __builtin_HEXAGON_M2_mmpyl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.s0
  __builtin_HEXAGON_M2_mmpyl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyl.s1
  __builtin_HEXAGON_M2_mmpyuh_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.rs0
  __builtin_HEXAGON_M2_mmpyuh_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.rs1
  __builtin_HEXAGON_M2_mmpyuh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.s0
  __builtin_HEXAGON_M2_mmpyuh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyuh.s1
  __builtin_HEXAGON_M2_mmpyul_rs0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.rs0
  __builtin_HEXAGON_M2_mmpyul_rs1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.rs1
  __builtin_HEXAGON_M2_mmpyul_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.s0
  __builtin_HEXAGON_M2_mmpyul_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mmpyul.s1
  __builtin_HEXAGON_M2_mpy_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hh.s0
  __builtin_HEXAGON_M2_mpy_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hh.s1
  __builtin_HEXAGON_M2_mpy_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hl.s0
  __builtin_HEXAGON_M2_mpy_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.hl.s1
  __builtin_HEXAGON_M2_mpy_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.lh.s0
  __builtin_HEXAGON_M2_mpy_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.lh.s1
  __builtin_HEXAGON_M2_mpy_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.ll.s0
  __builtin_HEXAGON_M2_mpy_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.ll.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hh.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hh.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hl.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.hl.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.lh.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.lh.s1
  __builtin_HEXAGON_M2_mpy_acc_sat_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.ll.s0
  __builtin_HEXAGON_M2_mpy_acc_sat_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.acc.sat.ll.s1
  __builtin_HEXAGON_M2_mpyd_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hh.s0
  __builtin_HEXAGON_M2_mpyd_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hh.s1
  __builtin_HEXAGON_M2_mpyd_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hl.s0
  __builtin_HEXAGON_M2_mpyd_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.hl.s1
  __builtin_HEXAGON_M2_mpyd_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.lh.s0
  __builtin_HEXAGON_M2_mpyd_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.lh.s1
  __builtin_HEXAGON_M2_mpyd_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.ll.s0
  __builtin_HEXAGON_M2_mpyd_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.acc.ll.s1
  __builtin_HEXAGON_M2_mpyd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hh.s0
  __builtin_HEXAGON_M2_mpyd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hh.s1
  __builtin_HEXAGON_M2_mpyd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hl.s0
  __builtin_HEXAGON_M2_mpyd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.hl.s1
  __builtin_HEXAGON_M2_mpyd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.lh.s0
  __builtin_HEXAGON_M2_mpyd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.lh.s1
  __builtin_HEXAGON_M2_mpyd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.ll.s0
  __builtin_HEXAGON_M2_mpyd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.ll.s1
  __builtin_HEXAGON_M2_mpyd_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hh.s0
  __builtin_HEXAGON_M2_mpyd_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hh.s1
  __builtin_HEXAGON_M2_mpyd_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hl.s0
  __builtin_HEXAGON_M2_mpyd_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.hl.s1
  __builtin_HEXAGON_M2_mpyd_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.lh.s0
  __builtin_HEXAGON_M2_mpyd_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.lh.s1
  __builtin_HEXAGON_M2_mpyd_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.ll.s0
  __builtin_HEXAGON_M2_mpyd_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.nac.ll.s1
  __builtin_HEXAGON_M2_mpyd_rnd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hh.s0
  __builtin_HEXAGON_M2_mpyd_rnd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hh.s1
  __builtin_HEXAGON_M2_mpyd_rnd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hl.s0
  __builtin_HEXAGON_M2_mpyd_rnd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.hl.s1
  __builtin_HEXAGON_M2_mpyd_rnd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.lh.s0
  __builtin_HEXAGON_M2_mpyd_rnd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.lh.s1
  __builtin_HEXAGON_M2_mpyd_rnd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.ll.s0
  __builtin_HEXAGON_M2_mpyd_rnd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyd.rnd.ll.s1
  __builtin_HEXAGON_M2_mpy_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hh.s0
  __builtin_HEXAGON_M2_mpy_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hh.s1
  __builtin_HEXAGON_M2_mpy_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hl.s0
  __builtin_HEXAGON_M2_mpy_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.hl.s1
  __builtin_HEXAGON_M2_mpyi(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyi
  __builtin_HEXAGON_M2_mpy_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.lh.s0
  __builtin_HEXAGON_M2_mpy_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.lh.s1
  __builtin_HEXAGON_M2_mpy_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.ll.s0
  __builtin_HEXAGON_M2_mpy_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.ll.s1
  __builtin_HEXAGON_M2_mpy_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hh.s0
  __builtin_HEXAGON_M2_mpy_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hh.s1
  __builtin_HEXAGON_M2_mpy_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hl.s0
  __builtin_HEXAGON_M2_mpy_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.hl.s1
  __builtin_HEXAGON_M2_mpy_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.lh.s0
  __builtin_HEXAGON_M2_mpy_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.lh.s1
  __builtin_HEXAGON_M2_mpy_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.ll.s0
  __builtin_HEXAGON_M2_mpy_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.ll.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hh.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hh.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hl.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.hl.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.lh.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.lh.s1
  __builtin_HEXAGON_M2_mpy_nac_sat_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.ll.s0
  __builtin_HEXAGON_M2_mpy_nac_sat_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.nac.sat.ll.s1
  __builtin_HEXAGON_M2_mpy_rnd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hh.s0
  __builtin_HEXAGON_M2_mpy_rnd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hh.s1
  __builtin_HEXAGON_M2_mpy_rnd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hl.s0
  __builtin_HEXAGON_M2_mpy_rnd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.hl.s1
  __builtin_HEXAGON_M2_mpy_rnd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.lh.s0
  __builtin_HEXAGON_M2_mpy_rnd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.lh.s1
  __builtin_HEXAGON_M2_mpy_rnd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.ll.s0
  __builtin_HEXAGON_M2_mpy_rnd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.rnd.ll.s1
  __builtin_HEXAGON_M2_mpy_sat_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hh.s0
  __builtin_HEXAGON_M2_mpy_sat_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hh.s1
  __builtin_HEXAGON_M2_mpy_sat_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hl.s0
  __builtin_HEXAGON_M2_mpy_sat_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.hl.s1
  __builtin_HEXAGON_M2_mpy_sat_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.lh.s0
  __builtin_HEXAGON_M2_mpy_sat_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.lh.s1
  __builtin_HEXAGON_M2_mpy_sat_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.ll.s0
  __builtin_HEXAGON_M2_mpy_sat_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.ll.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hh.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hh.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hl.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.hl.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.lh.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.lh.s1
  __builtin_HEXAGON_M2_mpy_sat_rnd_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.ll.s0
  __builtin_HEXAGON_M2_mpy_sat_rnd_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.sat.rnd.ll.s1
  __builtin_HEXAGON_M2_mpysmi(0, 0);
  // CHECK: @llvm.hexagon.M2.mpysmi
  __builtin_HEXAGON_M2_mpysu_up(0, 0);
  // CHECK: @llvm.hexagon.M2.mpysu.up
  __builtin_HEXAGON_M2_mpyu_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hh.s0
  __builtin_HEXAGON_M2_mpyu_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hh.s1
  __builtin_HEXAGON_M2_mpyu_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hl.s0
  __builtin_HEXAGON_M2_mpyu_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.hl.s1
  __builtin_HEXAGON_M2_mpyu_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.lh.s0
  __builtin_HEXAGON_M2_mpyu_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.lh.s1
  __builtin_HEXAGON_M2_mpyu_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.ll.s0
  __builtin_HEXAGON_M2_mpyu_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.acc.ll.s1
  __builtin_HEXAGON_M2_mpyud_acc_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hh.s0
  __builtin_HEXAGON_M2_mpyud_acc_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hh.s1
  __builtin_HEXAGON_M2_mpyud_acc_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hl.s0
  __builtin_HEXAGON_M2_mpyud_acc_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.hl.s1
  __builtin_HEXAGON_M2_mpyud_acc_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.lh.s0
  __builtin_HEXAGON_M2_mpyud_acc_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.lh.s1
  __builtin_HEXAGON_M2_mpyud_acc_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.ll.s0
  __builtin_HEXAGON_M2_mpyud_acc_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.acc.ll.s1
  __builtin_HEXAGON_M2_mpyud_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hh.s0
  __builtin_HEXAGON_M2_mpyud_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hh.s1
  __builtin_HEXAGON_M2_mpyud_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hl.s0
  __builtin_HEXAGON_M2_mpyud_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.hl.s1
  __builtin_HEXAGON_M2_mpyud_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.lh.s0
  __builtin_HEXAGON_M2_mpyud_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.lh.s1
  __builtin_HEXAGON_M2_mpyud_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.ll.s0
  __builtin_HEXAGON_M2_mpyud_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.ll.s1
  __builtin_HEXAGON_M2_mpyud_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hh.s0
  __builtin_HEXAGON_M2_mpyud_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hh.s1
  __builtin_HEXAGON_M2_mpyud_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hl.s0
  __builtin_HEXAGON_M2_mpyud_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.hl.s1
  __builtin_HEXAGON_M2_mpyud_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.lh.s0
  __builtin_HEXAGON_M2_mpyud_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.lh.s1
  __builtin_HEXAGON_M2_mpyud_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.ll.s0
  __builtin_HEXAGON_M2_mpyud_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyud.nac.ll.s1
  __builtin_HEXAGON_M2_mpyu_hh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hh.s0
  __builtin_HEXAGON_M2_mpyu_hh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hh.s1
  __builtin_HEXAGON_M2_mpyu_hl_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hl.s0
  __builtin_HEXAGON_M2_mpyu_hl_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.hl.s1
  __builtin_HEXAGON_M2_mpyui(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyui
  __builtin_HEXAGON_M2_mpyu_lh_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.lh.s0
  __builtin_HEXAGON_M2_mpyu_lh_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.lh.s1
  __builtin_HEXAGON_M2_mpyu_ll_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.ll.s0
  __builtin_HEXAGON_M2_mpyu_ll_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.ll.s1
  __builtin_HEXAGON_M2_mpyu_nac_hh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hh.s0
  __builtin_HEXAGON_M2_mpyu_nac_hh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hh.s1
  __builtin_HEXAGON_M2_mpyu_nac_hl_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hl.s0
  __builtin_HEXAGON_M2_mpyu_nac_hl_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.hl.s1
  __builtin_HEXAGON_M2_mpyu_nac_lh_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.lh.s0
  __builtin_HEXAGON_M2_mpyu_nac_lh_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.lh.s1
  __builtin_HEXAGON_M2_mpyu_nac_ll_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.ll.s0
  __builtin_HEXAGON_M2_mpyu_nac_ll_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.nac.ll.s1
  __builtin_HEXAGON_M2_mpy_up(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.up
  __builtin_HEXAGON_M2_mpy_up_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.up.s1
  __builtin_HEXAGON_M2_mpy_up_s1_sat(0, 0);
  // CHECK: @llvm.hexagon.M2.mpy.up.s1.sat
  __builtin_HEXAGON_M2_mpyu_up(0, 0);
  // CHECK: @llvm.hexagon.M2.mpyu.up
  __builtin_HEXAGON_M2_nacci(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.nacci
  __builtin_HEXAGON_M2_naccii(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.naccii
  __builtin_HEXAGON_M2_subacc(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.subacc
  __builtin_HEXAGON_M2_vabsdiffh(0, 0);
  // CHECK: @llvm.hexagon.M2.vabsdiffh
  __builtin_HEXAGON_M2_vabsdiffw(0, 0);
  // CHECK: @llvm.hexagon.M2.vabsdiffw
  __builtin_HEXAGON_M2_vcmac_s0_sat_i(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vcmac.s0.sat.i
  __builtin_HEXAGON_M2_vcmac_s0_sat_r(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vcmac.s0.sat.r
  __builtin_HEXAGON_M2_vcmpy_s0_sat_i(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s0.sat.i
  __builtin_HEXAGON_M2_vcmpy_s0_sat_r(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s0.sat.r
  __builtin_HEXAGON_M2_vcmpy_s1_sat_i(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s1.sat.i
  __builtin_HEXAGON_M2_vcmpy_s1_sat_r(0, 0);
  // CHECK: @llvm.hexagon.M2.vcmpy.s1.sat.r
  __builtin_HEXAGON_M2_vdmacs_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vdmacs.s0
  __builtin_HEXAGON_M2_vdmacs_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vdmacs.s1
  __builtin_HEXAGON_M2_vdmpyrs_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpyrs.s0
  __builtin_HEXAGON_M2_vdmpyrs_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpyrs.s1
  __builtin_HEXAGON_M2_vdmpys_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpys.s0
  __builtin_HEXAGON_M2_vdmpys_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vdmpys.s1
  __builtin_HEXAGON_M2_vmac2(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2
  __builtin_HEXAGON_M2_vmac2es(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2es
  __builtin_HEXAGON_M2_vmac2es_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2es.s0
  __builtin_HEXAGON_M2_vmac2es_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2es.s1
  __builtin_HEXAGON_M2_vmac2s_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2s.s0
  __builtin_HEXAGON_M2_vmac2s_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2s.s1
  __builtin_HEXAGON_M2_vmac2su_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2su.s0
  __builtin_HEXAGON_M2_vmac2su_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vmac2su.s1
  __builtin_HEXAGON_M2_vmpy2es_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2es.s0
  __builtin_HEXAGON_M2_vmpy2es_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2es.s1
  __builtin_HEXAGON_M2_vmpy2s_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s0
  __builtin_HEXAGON_M2_vmpy2s_s0pack(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s0pack
  __builtin_HEXAGON_M2_vmpy2s_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s1
  __builtin_HEXAGON_M2_vmpy2s_s1pack(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2s.s1pack
  __builtin_HEXAGON_M2_vmpy2su_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2su.s0
  __builtin_HEXAGON_M2_vmpy2su_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vmpy2su.s1
  __builtin_HEXAGON_M2_vraddh(0, 0);
  // CHECK: @llvm.hexagon.M2.vraddh
  __builtin_HEXAGON_M2_vradduh(0, 0);
  // CHECK: @llvm.hexagon.M2.vradduh
  __builtin_HEXAGON_M2_vrcmaci_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmaci.s0
  __builtin_HEXAGON_M2_vrcmaci_s0c(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmaci.s0c
  __builtin_HEXAGON_M2_vrcmacr_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmacr.s0
  __builtin_HEXAGON_M2_vrcmacr_s0c(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmacr.s0c
  __builtin_HEXAGON_M2_vrcmpyi_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyi.s0
  __builtin_HEXAGON_M2_vrcmpyi_s0c(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyi.s0c
  __builtin_HEXAGON_M2_vrcmpyr_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyr.s0
  __builtin_HEXAGON_M2_vrcmpyr_s0c(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpyr.s0c
  __builtin_HEXAGON_M2_vrcmpys_acc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpys.acc.s1
  __builtin_HEXAGON_M2_vrcmpys_s1(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpys.s1
  __builtin_HEXAGON_M2_vrcmpys_s1rp(0, 0);
  // CHECK: @llvm.hexagon.M2.vrcmpys.s1rp
  __builtin_HEXAGON_M2_vrmac_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.vrmac.s0
  __builtin_HEXAGON_M2_vrmpy_s0(0, 0);
  // CHECK: @llvm.hexagon.M2.vrmpy.s0
  __builtin_HEXAGON_M2_xor_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.M2.xor.xacc
  __builtin_HEXAGON_M4_and_and(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.and
  __builtin_HEXAGON_M4_and_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.andn
  __builtin_HEXAGON_M4_and_or(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.or
  __builtin_HEXAGON_M4_and_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.and.xor
  __builtin_HEXAGON_M4_cmpyi_wh(0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyi.wh
  __builtin_HEXAGON_M4_cmpyi_whc(0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyi.whc
  __builtin_HEXAGON_M4_cmpyr_wh(0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyr.wh
  __builtin_HEXAGON_M4_cmpyr_whc(0, 0);
  // CHECK: @llvm.hexagon.M4.cmpyr.whc
  __builtin_HEXAGON_M4_mac_up_s1_sat(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mac.up.s1.sat
  __builtin_HEXAGON_M4_mpyri_addi(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyri.addi
  __builtin_HEXAGON_M4_mpyri_addr(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyri.addr
  __builtin_HEXAGON_M4_mpyri_addr_u2(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyri.addr.u2
  __builtin_HEXAGON_M4_mpyrr_addi(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyrr.addi
  __builtin_HEXAGON_M4_mpyrr_addr(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.mpyrr.addr
  __builtin_HEXAGON_M4_nac_up_s1_sat(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.nac.up.s1.sat
  __builtin_HEXAGON_M4_or_and(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.and
  __builtin_HEXAGON_M4_or_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.andn
  __builtin_HEXAGON_M4_or_or(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.or
  __builtin_HEXAGON_M4_or_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.or.xor
  __builtin_HEXAGON_M4_pmpyw(0, 0);
  // CHECK: @llvm.hexagon.M4.pmpyw
  __builtin_HEXAGON_M4_pmpyw_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.pmpyw.acc
  __builtin_HEXAGON_M4_vpmpyh(0, 0);
  // CHECK: @llvm.hexagon.M4.vpmpyh
  __builtin_HEXAGON_M4_vpmpyh_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vpmpyh.acc
  __builtin_HEXAGON_M4_vrmpyeh_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.acc.s0
  __builtin_HEXAGON_M4_vrmpyeh_acc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.acc.s1
  __builtin_HEXAGON_M4_vrmpyeh_s0(0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.s0
  __builtin_HEXAGON_M4_vrmpyeh_s1(0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyeh.s1
  __builtin_HEXAGON_M4_vrmpyoh_acc_s0(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.acc.s0
  __builtin_HEXAGON_M4_vrmpyoh_acc_s1(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.acc.s1
  __builtin_HEXAGON_M4_vrmpyoh_s0(0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.s0
  __builtin_HEXAGON_M4_vrmpyoh_s1(0, 0);
  // CHECK: @llvm.hexagon.M4.vrmpyoh.s1
  __builtin_HEXAGON_M4_xor_and(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.xor.and
  __builtin_HEXAGON_M4_xor_andn(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.xor.andn
  __builtin_HEXAGON_M4_xor_or(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.xor.or
  __builtin_HEXAGON_M4_xor_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.M4.xor.xacc
  __builtin_HEXAGON_M5_vdmacbsu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vdmacbsu
  __builtin_HEXAGON_M5_vdmpybsu(0, 0);
  // CHECK: @llvm.hexagon.M5.vdmpybsu
  __builtin_HEXAGON_M5_vmacbsu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vmacbsu
  __builtin_HEXAGON_M5_vmacbuu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vmacbuu
  __builtin_HEXAGON_M5_vmpybsu(0, 0);
  // CHECK: @llvm.hexagon.M5.vmpybsu
  __builtin_HEXAGON_M5_vmpybuu(0, 0);
  // CHECK: @llvm.hexagon.M5.vmpybuu
  __builtin_HEXAGON_M5_vrmacbsu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vrmacbsu
  __builtin_HEXAGON_M5_vrmacbuu(0, 0, 0);
  // CHECK: @llvm.hexagon.M5.vrmacbuu
  __builtin_HEXAGON_M5_vrmpybsu(0, 0);
  // CHECK: @llvm.hexagon.M5.vrmpybsu
  __builtin_HEXAGON_M5_vrmpybuu(0, 0);
  // CHECK: @llvm.hexagon.M5.vrmpybuu
  __builtin_HEXAGON_S2_addasl_rrri(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.addasl.rrri
  __builtin_HEXAGON_S2_asl_i_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p
  __builtin_HEXAGON_S2_asl_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.acc
  __builtin_HEXAGON_S2_asl_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.and
  __builtin_HEXAGON_S2_asl_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.nac
  __builtin_HEXAGON_S2_asl_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.or
  __builtin_HEXAGON_S2_asl_i_p_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.p.xacc
  __builtin_HEXAGON_S2_asl_i_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r
  __builtin_HEXAGON_S2_asl_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.acc
  __builtin_HEXAGON_S2_asl_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.and
  __builtin_HEXAGON_S2_asl_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.nac
  __builtin_HEXAGON_S2_asl_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.or
  __builtin_HEXAGON_S2_asl_i_r_sat(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.sat
  __builtin_HEXAGON_S2_asl_i_r_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.r.xacc
  __builtin_HEXAGON_S2_asl_i_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.vh
  __builtin_HEXAGON_S2_asl_i_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.i.vw
  __builtin_HEXAGON_S2_asl_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p
  __builtin_HEXAGON_S2_asl_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.acc
  __builtin_HEXAGON_S2_asl_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.and
  __builtin_HEXAGON_S2_asl_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.nac
  __builtin_HEXAGON_S2_asl_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.or
  __builtin_HEXAGON_S2_asl_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.p.xor
  __builtin_HEXAGON_S2_asl_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r
  __builtin_HEXAGON_S2_asl_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.acc
  __builtin_HEXAGON_S2_asl_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.and
  __builtin_HEXAGON_S2_asl_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.nac
  __builtin_HEXAGON_S2_asl_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.or
  __builtin_HEXAGON_S2_asl_r_r_sat(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.r.sat
  __builtin_HEXAGON_S2_asl_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.vh
  __builtin_HEXAGON_S2_asl_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.asl.r.vw
  __builtin_HEXAGON_S2_asr_i_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p
  __builtin_HEXAGON_S2_asr_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.acc
  __builtin_HEXAGON_S2_asr_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.and
  __builtin_HEXAGON_S2_asr_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.nac
  __builtin_HEXAGON_S2_asr_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.or
  __builtin_HEXAGON_S2_asr_i_p_rnd(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.rnd
  __builtin_HEXAGON_S2_asr_i_p_rnd_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax
  __builtin_HEXAGON_S2_asr_i_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r
  __builtin_HEXAGON_S2_asr_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.acc
  __builtin_HEXAGON_S2_asr_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.and
  __builtin_HEXAGON_S2_asr_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.nac
  __builtin_HEXAGON_S2_asr_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.or
  __builtin_HEXAGON_S2_asr_i_r_rnd(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.rnd
  __builtin_HEXAGON_S2_asr_i_r_rnd_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.r.rnd.goodsyntax
  __builtin_HEXAGON_S2_asr_i_svw_trun(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.svw.trun
  __builtin_HEXAGON_S2_asr_i_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.vh
  __builtin_HEXAGON_S2_asr_i_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.i.vw
  __builtin_HEXAGON_S2_asr_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p
  __builtin_HEXAGON_S2_asr_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.acc
  __builtin_HEXAGON_S2_asr_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.and
  __builtin_HEXAGON_S2_asr_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.nac
  __builtin_HEXAGON_S2_asr_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.or
  __builtin_HEXAGON_S2_asr_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.p.xor
  __builtin_HEXAGON_S2_asr_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r
  __builtin_HEXAGON_S2_asr_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.acc
  __builtin_HEXAGON_S2_asr_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.and
  __builtin_HEXAGON_S2_asr_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.nac
  __builtin_HEXAGON_S2_asr_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.or
  __builtin_HEXAGON_S2_asr_r_r_sat(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.r.sat
  __builtin_HEXAGON_S2_asr_r_svw_trun(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.svw.trun
  __builtin_HEXAGON_S2_asr_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.vh
  __builtin_HEXAGON_S2_asr_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.asr.r.vw
  __builtin_HEXAGON_S2_brev(0);
  // CHECK: @llvm.hexagon.S2.brev
  __builtin_HEXAGON_S2_brevp(0);
  // CHECK: @llvm.hexagon.S2.brevp
  __builtin_HEXAGON_S2_cabacencbin(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.cabacencbin
  __builtin_HEXAGON_S2_cl0(0);
  // CHECK: @llvm.hexagon.S2.cl0
  __builtin_HEXAGON_S2_cl0p(0);
  // CHECK: @llvm.hexagon.S2.cl0p
  __builtin_HEXAGON_S2_cl1(0);
  // CHECK: @llvm.hexagon.S2.cl1
  __builtin_HEXAGON_S2_cl1p(0);
  // CHECK: @llvm.hexagon.S2.cl1p
  __builtin_HEXAGON_S2_clb(0);
  // CHECK: @llvm.hexagon.S2.clb
  __builtin_HEXAGON_S2_clbnorm(0);
  // CHECK: @llvm.hexagon.S2.clbnorm
  __builtin_HEXAGON_S2_clbp(0);
  // CHECK: @llvm.hexagon.S2.clbp
  __builtin_HEXAGON_S2_clrbit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.clrbit.i
  __builtin_HEXAGON_S2_clrbit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.clrbit.r
  __builtin_HEXAGON_S2_ct0(0);
  // CHECK: @llvm.hexagon.S2.ct0
  __builtin_HEXAGON_S2_ct0p(0);
  // CHECK: @llvm.hexagon.S2.ct0p
  __builtin_HEXAGON_S2_ct1(0);
  // CHECK: @llvm.hexagon.S2.ct1
  __builtin_HEXAGON_S2_ct1p(0);
  // CHECK: @llvm.hexagon.S2.ct1p
  __builtin_HEXAGON_S2_deinterleave(0);
  // CHECK: @llvm.hexagon.S2.deinterleave
  __builtin_HEXAGON_S2_extractu(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.extractu
  __builtin_HEXAGON_S2_extractup(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.extractup
  __builtin_HEXAGON_S2_extractup_rp(0, 0);
  // CHECK: @llvm.hexagon.S2.extractup.rp
  __builtin_HEXAGON_S2_extractu_rp(0, 0);
  // CHECK: @llvm.hexagon.S2.extractu.rp
  __builtin_HEXAGON_S2_insert(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.insert
  __builtin_HEXAGON_S2_insertp(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.insertp
  __builtin_HEXAGON_S2_insertp_rp(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.insertp.rp
  __builtin_HEXAGON_S2_insert_rp(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.insert.rp
  __builtin_HEXAGON_S2_interleave(0);
  // CHECK: @llvm.hexagon.S2.interleave
  __builtin_HEXAGON_S2_lfsp(0, 0);
  // CHECK: @llvm.hexagon.S2.lfsp
  __builtin_HEXAGON_S2_lsl_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p
  __builtin_HEXAGON_S2_lsl_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.acc
  __builtin_HEXAGON_S2_lsl_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.and
  __builtin_HEXAGON_S2_lsl_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.nac
  __builtin_HEXAGON_S2_lsl_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.or
  __builtin_HEXAGON_S2_lsl_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.p.xor
  __builtin_HEXAGON_S2_lsl_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r
  __builtin_HEXAGON_S2_lsl_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.acc
  __builtin_HEXAGON_S2_lsl_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.and
  __builtin_HEXAGON_S2_lsl_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.nac
  __builtin_HEXAGON_S2_lsl_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.r.or
  __builtin_HEXAGON_S2_lsl_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.vh
  __builtin_HEXAGON_S2_lsl_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.lsl.r.vw
  __builtin_HEXAGON_S2_lsr_i_p(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p
  __builtin_HEXAGON_S2_lsr_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.acc
  __builtin_HEXAGON_S2_lsr_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.and
  __builtin_HEXAGON_S2_lsr_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.nac
  __builtin_HEXAGON_S2_lsr_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.or
  __builtin_HEXAGON_S2_lsr_i_p_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.p.xacc
  __builtin_HEXAGON_S2_lsr_i_r(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r
  __builtin_HEXAGON_S2_lsr_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.acc
  __builtin_HEXAGON_S2_lsr_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.and
  __builtin_HEXAGON_S2_lsr_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.nac
  __builtin_HEXAGON_S2_lsr_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.or
  __builtin_HEXAGON_S2_lsr_i_r_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.r.xacc
  __builtin_HEXAGON_S2_lsr_i_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.vh
  __builtin_HEXAGON_S2_lsr_i_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.i.vw
  __builtin_HEXAGON_S2_lsr_r_p(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p
  __builtin_HEXAGON_S2_lsr_r_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.acc
  __builtin_HEXAGON_S2_lsr_r_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.and
  __builtin_HEXAGON_S2_lsr_r_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.nac
  __builtin_HEXAGON_S2_lsr_r_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.or
  __builtin_HEXAGON_S2_lsr_r_p_xor(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.p.xor
  __builtin_HEXAGON_S2_lsr_r_r(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r
  __builtin_HEXAGON_S2_lsr_r_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.acc
  __builtin_HEXAGON_S2_lsr_r_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.and
  __builtin_HEXAGON_S2_lsr_r_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.nac
  __builtin_HEXAGON_S2_lsr_r_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.r.or
  __builtin_HEXAGON_S2_lsr_r_vh(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.vh
  __builtin_HEXAGON_S2_lsr_r_vw(0, 0);
  // CHECK: @llvm.hexagon.S2.lsr.r.vw
  __builtin_HEXAGON_S2_packhl(0, 0);
  // CHECK: @llvm.hexagon.S2.packhl
  __builtin_HEXAGON_S2_parityp(0, 0);
  // CHECK: @llvm.hexagon.S2.parityp
  __builtin_HEXAGON_S2_setbit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.setbit.i
  __builtin_HEXAGON_S2_setbit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.setbit.r
  __builtin_HEXAGON_S2_shuffeb(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffeb
  __builtin_HEXAGON_S2_shuffeh(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffeh
  __builtin_HEXAGON_S2_shuffob(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffob
  __builtin_HEXAGON_S2_shuffoh(0, 0);
  // CHECK: @llvm.hexagon.S2.shuffoh
  __builtin_HEXAGON_S2_svsathb(0);
  // CHECK: @llvm.hexagon.S2.svsathb
  __builtin_HEXAGON_S2_svsathub(0);
  // CHECK: @llvm.hexagon.S2.svsathub
  __builtin_HEXAGON_S2_tableidxb_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.tableidxb.goodsyntax
  __builtin_HEXAGON_S2_tableidxd_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.tableidxd.goodsyntax
  __builtin_HEXAGON_S2_tableidxh_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.tableidxh.goodsyntax
  __builtin_HEXAGON_S2_tableidxw_goodsyntax(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S2.tableidxw.goodsyntax
  __builtin_HEXAGON_S2_togglebit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.togglebit.i
  __builtin_HEXAGON_S2_togglebit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.togglebit.r
  __builtin_HEXAGON_S2_tstbit_i(0, 0);
  // CHECK: @llvm.hexagon.S2.tstbit.i
  __builtin_HEXAGON_S2_tstbit_r(0, 0);
  // CHECK: @llvm.hexagon.S2.tstbit.r
  __builtin_HEXAGON_S2_valignib(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.valignib
  __builtin_HEXAGON_S2_valignrb(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.valignrb
  __builtin_HEXAGON_S2_vcnegh(0, 0);
  // CHECK: @llvm.hexagon.S2.vcnegh
  __builtin_HEXAGON_S2_vcrotate(0, 0);
  // CHECK: @llvm.hexagon.S2.vcrotate
  __builtin_HEXAGON_S2_vrcnegh(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.vrcnegh
  __builtin_HEXAGON_S2_vrndpackwh(0);
  // CHECK: @llvm.hexagon.S2.vrndpackwh
  __builtin_HEXAGON_S2_vrndpackwhs(0);
  // CHECK: @llvm.hexagon.S2.vrndpackwhs
  __builtin_HEXAGON_S2_vsathb(0);
  // CHECK: @llvm.hexagon.S2.vsathb
  __builtin_HEXAGON_S2_vsathb_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsathb.nopack
  __builtin_HEXAGON_S2_vsathub(0);
  // CHECK: @llvm.hexagon.S2.vsathub
  __builtin_HEXAGON_S2_vsathub_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsathub.nopack
  __builtin_HEXAGON_S2_vsatwh(0);
  // CHECK: @llvm.hexagon.S2.vsatwh
  __builtin_HEXAGON_S2_vsatwh_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsatwh.nopack
  __builtin_HEXAGON_S2_vsatwuh(0);
  // CHECK: @llvm.hexagon.S2.vsatwuh
  __builtin_HEXAGON_S2_vsatwuh_nopack(0);
  // CHECK: @llvm.hexagon.S2.vsatwuh.nopack
  __builtin_HEXAGON_S2_vsplatrb(0);
  // CHECK: @llvm.hexagon.S2.vsplatrb
  __builtin_HEXAGON_S2_vsplatrh(0);
  // CHECK: @llvm.hexagon.S2.vsplatrh
  __builtin_HEXAGON_S2_vspliceib(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.vspliceib
  __builtin_HEXAGON_S2_vsplicerb(0, 0, 0);
  // CHECK: @llvm.hexagon.S2.vsplicerb
  __builtin_HEXAGON_S2_vsxtbh(0);
  // CHECK: @llvm.hexagon.S2.vsxtbh
  __builtin_HEXAGON_S2_vsxthw(0);
  // CHECK: @llvm.hexagon.S2.vsxthw
  __builtin_HEXAGON_S2_vtrunehb(0);
  // CHECK: @llvm.hexagon.S2.vtrunehb
  __builtin_HEXAGON_S2_vtrunewh(0, 0);
  // CHECK: @llvm.hexagon.S2.vtrunewh
  __builtin_HEXAGON_S2_vtrunohb(0);
  // CHECK: @llvm.hexagon.S2.vtrunohb
  __builtin_HEXAGON_S2_vtrunowh(0, 0);
  // CHECK: @llvm.hexagon.S2.vtrunowh
  __builtin_HEXAGON_S2_vzxtbh(0);
  // CHECK: @llvm.hexagon.S2.vzxtbh
  __builtin_HEXAGON_S2_vzxthw(0);
  // CHECK: @llvm.hexagon.S2.vzxthw
  __builtin_HEXAGON_S4_addaddi(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.addaddi
  __builtin_HEXAGON_S4_addi_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.addi.asl.ri
  __builtin_HEXAGON_S4_addi_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.addi.lsr.ri
  __builtin_HEXAGON_S4_andi_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.andi.asl.ri
  __builtin_HEXAGON_S4_andi_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.andi.lsr.ri
  __builtin_HEXAGON_S4_clbaddi(0, 0);
  // CHECK: @llvm.hexagon.S4.clbaddi
  __builtin_HEXAGON_S4_clbpaddi(0, 0);
  // CHECK: @llvm.hexagon.S4.clbpaddi
  __builtin_HEXAGON_S4_clbpnorm(0);
  // CHECK: @llvm.hexagon.S4.clbpnorm
  __builtin_HEXAGON_S4_extract(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.extract
  __builtin_HEXAGON_S4_extractp(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.extractp
  __builtin_HEXAGON_S4_extractp_rp(0, 0);
  // CHECK: @llvm.hexagon.S4.extractp.rp
  __builtin_HEXAGON_S4_extract_rp(0, 0);
  // CHECK: @llvm.hexagon.S4.extract.rp
  __builtin_HEXAGON_S4_lsli(0, 0);
  // CHECK: @llvm.hexagon.S4.lsli
  __builtin_HEXAGON_S4_ntstbit_i(0, 0);
  // CHECK: @llvm.hexagon.S4.ntstbit.i
  __builtin_HEXAGON_S4_ntstbit_r(0, 0);
  // CHECK: @llvm.hexagon.S4.ntstbit.r
  __builtin_HEXAGON_S4_or_andi(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.or.andi
  __builtin_HEXAGON_S4_or_andix(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.or.andix
  __builtin_HEXAGON_S4_ori_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.ori.asl.ri
  __builtin_HEXAGON_S4_ori_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.ori.lsr.ri
  __builtin_HEXAGON_S4_or_ori(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.or.ori
  __builtin_HEXAGON_S4_parity(0, 0);
  // CHECK: @llvm.hexagon.S4.parity
  __builtin_HEXAGON_S4_subaddi(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.subaddi
  __builtin_HEXAGON_S4_subi_asl_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.subi.asl.ri
  __builtin_HEXAGON_S4_subi_lsr_ri(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.subi.lsr.ri
  __builtin_HEXAGON_S4_vrcrotate(0, 0, 0);
  // CHECK: @llvm.hexagon.S4.vrcrotate
  __builtin_HEXAGON_S4_vrcrotate_acc(0, 0, 0, 0);
  // CHECK: @llvm.hexagon.S4.vrcrotate.acc
  __builtin_HEXAGON_S4_vxaddsubh(0, 0);
  // CHECK: @llvm.hexagon.S4.vxaddsubh
  __builtin_HEXAGON_S4_vxaddsubhr(0, 0);
  // CHECK: @llvm.hexagon.S4.vxaddsubhr
  __builtin_HEXAGON_S4_vxaddsubw(0, 0);
  // CHECK: @llvm.hexagon.S4.vxaddsubw
  __builtin_HEXAGON_S4_vxsubaddh(0, 0);
  // CHECK: @llvm.hexagon.S4.vxsubaddh
  __builtin_HEXAGON_S4_vxsubaddhr(0, 0);
  // CHECK: @llvm.hexagon.S4.vxsubaddhr
  __builtin_HEXAGON_S4_vxsubaddw(0, 0);
  // CHECK: @llvm.hexagon.S4.vxsubaddw
  __builtin_HEXAGON_S5_asrhub_rnd_sat_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax
  __builtin_HEXAGON_S5_asrhub_sat(0, 0);
  // CHECK: @llvm.hexagon.S5.asrhub.sat
  __builtin_HEXAGON_S5_popcountp(0);
  // CHECK: @llvm.hexagon.S5.popcountp
  __builtin_HEXAGON_S5_vasrhrnd_goodsyntax(0, 0);
  // CHECK: @llvm.hexagon.S5.vasrhrnd.goodsyntax
  __builtin_HEXAGON_S6_rol_i_p(0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p
  __builtin_HEXAGON_S6_rol_i_p_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.acc
  __builtin_HEXAGON_S6_rol_i_p_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.and
  __builtin_HEXAGON_S6_rol_i_p_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.nac
  __builtin_HEXAGON_S6_rol_i_p_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.or
  __builtin_HEXAGON_S6_rol_i_p_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.p.xacc
  __builtin_HEXAGON_S6_rol_i_r(0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r
  __builtin_HEXAGON_S6_rol_i_r_acc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.acc
  __builtin_HEXAGON_S6_rol_i_r_and(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.and
  __builtin_HEXAGON_S6_rol_i_r_nac(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.nac
  __builtin_HEXAGON_S6_rol_i_r_or(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.or
  __builtin_HEXAGON_S6_rol_i_r_xacc(0, 0, 0);
  // CHECK: @llvm.hexagon.S6.rol.i.r.xacc
  __builtin_HEXAGON_V6_extractw_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.extractw.128B
  __builtin_HEXAGON_V6_extractw(v16, 0);
  // CHECK: @llvm.hexagon.V6.extractw
  __builtin_HEXAGON_V6_hi_128B(v64);
  // CHECK: @llvm.hexagon.V6.hi.128B
  __builtin_HEXAGON_V6_hi(v32);
  // CHECK: @llvm.hexagon.V6.hi
  __builtin_HEXAGON_V6_lo_128B(v64);
  // CHECK: @llvm.hexagon.V6.lo.128B
  __builtin_HEXAGON_V6_lo(v32);
  // CHECK: @llvm.hexagon.V6.lo
  __builtin_HEXAGON_V6_lvsplatw(0);
  // CHECK: @llvm.hexagon.V6.lvsplatw
  __builtin_HEXAGON_V6_lvsplatw_128B(0);
  // CHECK: @llvm.hexagon.V6.lvsplatw.128B
  __builtin_HEXAGON_V6_pred_and_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.pred.and.128B
  __builtin_HEXAGON_V6_pred_and_n_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.pred.and.n.128B
  __builtin_HEXAGON_V6_pred_and_n(v16, v16);
  // CHECK: @llvm.hexagon.V6.pred.and.n
  __builtin_HEXAGON_V6_pred_and(v16, v16);
  // CHECK: @llvm.hexagon.V6.pred.and
  __builtin_HEXAGON_V6_pred_not_128B(v32);
  // CHECK: @llvm.hexagon.V6.pred.not.128B
  __builtin_HEXAGON_V6_pred_not(v16);
  // CHECK: @llvm.hexagon.V6.pred.not
  __builtin_HEXAGON_V6_pred_or_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.pred.or.128B
  __builtin_HEXAGON_V6_pred_or_n_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.pred.or.n.128B
  __builtin_HEXAGON_V6_pred_or_n(v16, v16);
  // CHECK: @llvm.hexagon.V6.pred.or.n
  __builtin_HEXAGON_V6_pred_or(v16, v16);
  // CHECK: @llvm.hexagon.V6.pred.or
  __builtin_HEXAGON_V6_pred_scalar2(0);
  // CHECK: @llvm.hexagon.V6.pred.scalar2
  __builtin_HEXAGON_V6_pred_scalar2_128B(0);
  // CHECK: @llvm.hexagon.V6.pred.scalar2.128B
  __builtin_HEXAGON_V6_pred_xor_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.pred.xor.128B
  __builtin_HEXAGON_V6_pred_xor(v16, v16);
  // CHECK: @llvm.hexagon.V6.pred.xor
  __builtin_HEXAGON_V6_vabsdiffh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vabsdiffh.128B
  __builtin_HEXAGON_V6_vabsdiffh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vabsdiffh
  __builtin_HEXAGON_V6_vabsdiffub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vabsdiffub.128B
  __builtin_HEXAGON_V6_vabsdiffub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vabsdiffub
  __builtin_HEXAGON_V6_vabsdiffuh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vabsdiffuh.128B
  __builtin_HEXAGON_V6_vabsdiffuh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vabsdiffuh
  __builtin_HEXAGON_V6_vabsdiffw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vabsdiffw.128B
  __builtin_HEXAGON_V6_vabsdiffw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vabsdiffw
  __builtin_HEXAGON_V6_vabsh_128B(v32);
  // CHECK: @llvm.hexagon.V6.vabsh.128B
  __builtin_HEXAGON_V6_vabsh_sat_128B(v32);
  // CHECK: @llvm.hexagon.V6.vabsh.sat.128B
  __builtin_HEXAGON_V6_vabsh_sat(v16);
  // CHECK: @llvm.hexagon.V6.vabsh.sat
  __builtin_HEXAGON_V6_vabsh(v16);
  // CHECK: @llvm.hexagon.V6.vabsh
  __builtin_HEXAGON_V6_vabsw_128B(v32);
  // CHECK: @llvm.hexagon.V6.vabsw.128B
  __builtin_HEXAGON_V6_vabsw_sat_128B(v32);
  // CHECK: @llvm.hexagon.V6.vabsw.sat.128B
  __builtin_HEXAGON_V6_vabsw_sat(v16);
  // CHECK: @llvm.hexagon.V6.vabsw.sat
  __builtin_HEXAGON_V6_vabsw(v16);
  // CHECK: @llvm.hexagon.V6.vabsw
  __builtin_HEXAGON_V6_vaddb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddb.128B
  __builtin_HEXAGON_V6_vaddb_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddb.dv.128B
  __builtin_HEXAGON_V6_vaddb_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddb.dv
  __builtin_HEXAGON_V6_vaddbnq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddbnq.128B
  __builtin_HEXAGON_V6_vaddbnq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddbnq
  __builtin_HEXAGON_V6_vaddbq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddbq.128B
  __builtin_HEXAGON_V6_vaddbq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddbq
  __builtin_HEXAGON_V6_vaddb(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddb
  __builtin_HEXAGON_V6_vaddh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddh.128B
  __builtin_HEXAGON_V6_vaddh_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddh.dv.128B
  __builtin_HEXAGON_V6_vaddh_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddh.dv
  __builtin_HEXAGON_V6_vaddhnq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddhnq.128B
  __builtin_HEXAGON_V6_vaddhnq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddhnq
  __builtin_HEXAGON_V6_vaddhq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddhq.128B
  __builtin_HEXAGON_V6_vaddhq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddhq
  __builtin_HEXAGON_V6_vaddhsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddhsat.128B
  __builtin_HEXAGON_V6_vaddhsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddhsat.dv.128B
  __builtin_HEXAGON_V6_vaddhsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddhsat.dv
  __builtin_HEXAGON_V6_vaddhsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddhsat
  __builtin_HEXAGON_V6_vaddh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddh
  __builtin_HEXAGON_V6_vaddhw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddhw.128B
  __builtin_HEXAGON_V6_vaddhw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddhw
  __builtin_HEXAGON_V6_vaddubh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddubh.128B
  __builtin_HEXAGON_V6_vaddubh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddubh
  __builtin_HEXAGON_V6_vaddubsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddubsat.128B
  __builtin_HEXAGON_V6_vaddubsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddubsat.dv.128B
  __builtin_HEXAGON_V6_vaddubsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddubsat.dv
  __builtin_HEXAGON_V6_vaddubsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddubsat
  __builtin_HEXAGON_V6_vadduhsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vadduhsat.128B
  __builtin_HEXAGON_V6_vadduhsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vadduhsat.dv.128B
  __builtin_HEXAGON_V6_vadduhsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vadduhsat.dv
  __builtin_HEXAGON_V6_vadduhsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vadduhsat
  __builtin_HEXAGON_V6_vadduhw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vadduhw.128B
  __builtin_HEXAGON_V6_vadduhw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vadduhw
  __builtin_HEXAGON_V6_vaddw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddw.128B
  __builtin_HEXAGON_V6_vaddw_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddw.dv.128B
  __builtin_HEXAGON_V6_vaddw_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddw.dv
  __builtin_HEXAGON_V6_vaddwnq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddwnq.128B
  __builtin_HEXAGON_V6_vaddwnq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddwnq
  __builtin_HEXAGON_V6_vaddwq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddwq.128B
  __builtin_HEXAGON_V6_vaddwq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddwq
  __builtin_HEXAGON_V6_vaddwsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddwsat.128B
  __builtin_HEXAGON_V6_vaddwsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vaddwsat.dv.128B
  __builtin_HEXAGON_V6_vaddwsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaddwsat.dv
  __builtin_HEXAGON_V6_vaddwsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddwsat
  __builtin_HEXAGON_V6_vaddw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaddw
  __builtin_HEXAGON_V6_valignb_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.valignb.128B
  __builtin_HEXAGON_V6_valignbi_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.valignbi.128B
  __builtin_HEXAGON_V6_valignbi(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.valignbi
  __builtin_HEXAGON_V6_valignb(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.valignb
  __builtin_HEXAGON_V6_vand_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vand.128B
  __builtin_HEXAGON_V6_vandqrt_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt.128B
  __builtin_HEXAGON_V6_vandqrt_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt.acc.128B
  __builtin_HEXAGON_V6_vandqrt_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt.acc
  __builtin_HEXAGON_V6_vandqrt(v16, 0);
  // CHECK: @llvm.hexagon.V6.vandqrt
  __builtin_HEXAGON_V6_vand(v16, v16);
  // CHECK: @llvm.hexagon.V6.vand
  __builtin_HEXAGON_V6_vandvrt_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vandvrt.128B
  __builtin_HEXAGON_V6_vandvrt_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vandvrt.acc.128B
  __builtin_HEXAGON_V6_vandvrt_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vandvrt.acc
  __builtin_HEXAGON_V6_vandvrt(v16, 0);
  // CHECK: @llvm.hexagon.V6.vandvrt
  __builtin_HEXAGON_V6_vaslh_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vaslh.128B
  __builtin_HEXAGON_V6_vaslhv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaslhv.128B
  __builtin_HEXAGON_V6_vaslh(v16, 0);
  // CHECK: @llvm.hexagon.V6.vaslh
  __builtin_HEXAGON_V6_vaslhv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaslhv
  __builtin_HEXAGON_V6_vaslw_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vaslw.128B
  __builtin_HEXAGON_V6_vaslw_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vaslw.acc.128B
  __builtin_HEXAGON_V6_vaslw_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vaslw.acc
  __builtin_HEXAGON_V6_vaslwv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vaslwv.128B
  __builtin_HEXAGON_V6_vaslw(v16, 0);
  // CHECK: @llvm.hexagon.V6.vaslw
  __builtin_HEXAGON_V6_vaslwv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vaslwv
  __builtin_HEXAGON_V6_vasrh_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrh.128B
  __builtin_HEXAGON_V6_vasrhbrndsat_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrhbrndsat.128B
  __builtin_HEXAGON_V6_vasrhbrndsat(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrhbrndsat
  __builtin_HEXAGON_V6_vasrhubrndsat_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubrndsat.128B
  __builtin_HEXAGON_V6_vasrhubrndsat(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubrndsat
  __builtin_HEXAGON_V6_vasrhubsat_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubsat.128B
  __builtin_HEXAGON_V6_vasrhubsat(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrhubsat
  __builtin_HEXAGON_V6_vasrhv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vasrhv.128B
  __builtin_HEXAGON_V6_vasrh(v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrh
  __builtin_HEXAGON_V6_vasrhv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vasrhv
  __builtin_HEXAGON_V6_vasrw_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrw.128B
  __builtin_HEXAGON_V6_vasrw_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrw.acc.128B
  __builtin_HEXAGON_V6_vasrw_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrw.acc
  __builtin_HEXAGON_V6_vasrwh_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrwh.128B
  __builtin_HEXAGON_V6_vasrwhrndsat_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhrndsat.128B
  __builtin_HEXAGON_V6_vasrwhrndsat(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhrndsat
  __builtin_HEXAGON_V6_vasrwhsat_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhsat.128B
  __builtin_HEXAGON_V6_vasrwhsat(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrwhsat
  __builtin_HEXAGON_V6_vasrwh(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrwh
  __builtin_HEXAGON_V6_vasrwuhsat_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vasrwuhsat.128B
  __builtin_HEXAGON_V6_vasrwuhsat(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrwuhsat
  __builtin_HEXAGON_V6_vasrwv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vasrwv.128B
  __builtin_HEXAGON_V6_vasrw(v16, 0);
  // CHECK: @llvm.hexagon.V6.vasrw
  __builtin_HEXAGON_V6_vasrwv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vasrwv
  __builtin_HEXAGON_V6_vassign_128B(v32);
  // CHECK: @llvm.hexagon.V6.vassign.128B
  __builtin_HEXAGON_V6_vassignp_128B(v64);
  // CHECK: @llvm.hexagon.V6.vassignp.128B
  __builtin_HEXAGON_V6_vassignp(v32);
  // CHECK: @llvm.hexagon.V6.vassignp
  __builtin_HEXAGON_V6_vassign(v16);
  // CHECK: @llvm.hexagon.V6.vassign
  __builtin_HEXAGON_V6_vavgh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavgh.128B
  __builtin_HEXAGON_V6_vavghrnd_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavghrnd.128B
  __builtin_HEXAGON_V6_vavghrnd(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavghrnd
  __builtin_HEXAGON_V6_vavgh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavgh
  __builtin_HEXAGON_V6_vavgub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavgub.128B
  __builtin_HEXAGON_V6_vavgubrnd_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavgubrnd.128B
  __builtin_HEXAGON_V6_vavgubrnd(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavgubrnd
  __builtin_HEXAGON_V6_vavgub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavgub
  __builtin_HEXAGON_V6_vavguh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavguh.128B
  __builtin_HEXAGON_V6_vavguhrnd_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavguhrnd.128B
  __builtin_HEXAGON_V6_vavguhrnd(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavguhrnd
  __builtin_HEXAGON_V6_vavguh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavguh
  __builtin_HEXAGON_V6_vavgw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavgw.128B
  __builtin_HEXAGON_V6_vavgwrnd_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vavgwrnd.128B
  __builtin_HEXAGON_V6_vavgwrnd(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavgwrnd
  __builtin_HEXAGON_V6_vavgw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vavgw
  __builtin_HEXAGON_V6_vcl0h_128B(v32);
  // CHECK: @llvm.hexagon.V6.vcl0h.128B
  __builtin_HEXAGON_V6_vcl0h(v16);
  // CHECK: @llvm.hexagon.V6.vcl0h
  __builtin_HEXAGON_V6_vcl0w_128B(v32);
  // CHECK: @llvm.hexagon.V6.vcl0w.128B
  __builtin_HEXAGON_V6_vcl0w(v16);
  // CHECK: @llvm.hexagon.V6.vcl0w
  __builtin_HEXAGON_V6_vcombine_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vcombine.128B
  __builtin_HEXAGON_V6_vcombine(v16, v16);
  // CHECK: @llvm.hexagon.V6.vcombine
  __builtin_HEXAGON_V6_vd0_128B();
  // CHECK: @llvm.hexagon.V6.vd0.128B
  __builtin_HEXAGON_V6_vd0();
  // CHECK: @llvm.hexagon.V6.vd0
  __builtin_HEXAGON_V6_vdealb_128B(v32);
  // CHECK: @llvm.hexagon.V6.vdealb.128B
  __builtin_HEXAGON_V6_vdealb4w_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vdealb4w.128B
  __builtin_HEXAGON_V6_vdealb4w(v16, v16);
  // CHECK: @llvm.hexagon.V6.vdealb4w
  __builtin_HEXAGON_V6_vdealb(v16);
  // CHECK: @llvm.hexagon.V6.vdealb
  __builtin_HEXAGON_V6_vdealh_128B(v32);
  // CHECK: @llvm.hexagon.V6.vdealh.128B
  __builtin_HEXAGON_V6_vdealh(v16);
  // CHECK: @llvm.hexagon.V6.vdealh
  __builtin_HEXAGON_V6_vdealvdd_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdealvdd.128B
  __builtin_HEXAGON_V6_vdealvdd(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vdealvdd
  __builtin_HEXAGON_V6_vdelta_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vdelta.128B
  __builtin_HEXAGON_V6_vdelta(v16, v16);
  // CHECK: @llvm.hexagon.V6.vdelta
  __builtin_HEXAGON_V6_vdmpybus_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.128B
  __builtin_HEXAGON_V6_vdmpybus_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.acc.128B
  __builtin_HEXAGON_V6_vdmpybus_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.acc
  __builtin_HEXAGON_V6_vdmpybus_dv_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv.128B
  __builtin_HEXAGON_V6_vdmpybus_dv_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv.acc.128B
  __builtin_HEXAGON_V6_vdmpybus_dv_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv.acc
  __builtin_HEXAGON_V6_vdmpybus_dv(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus.dv
  __builtin_HEXAGON_V6_vdmpybus(v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpybus
  __builtin_HEXAGON_V6_vdmpyhb_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.128B
  __builtin_HEXAGON_V6_vdmpyhb_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.acc.128B
  __builtin_HEXAGON_V6_vdmpyhb_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.acc
  __builtin_HEXAGON_V6_vdmpyhb_dv_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv.128B
  __builtin_HEXAGON_V6_vdmpyhb_dv_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv.acc.128B
  __builtin_HEXAGON_V6_vdmpyhb_dv_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv.acc
  __builtin_HEXAGON_V6_vdmpyhb_dv(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb.dv
  __builtin_HEXAGON_V6_vdmpyhb(v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhb
  __builtin_HEXAGON_V6_vdmpyhisat_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat.128B
  __builtin_HEXAGON_V6_vdmpyhisat_acc_128B(v32, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhisat_acc(v16, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat.acc
  __builtin_HEXAGON_V6_vdmpyhisat(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhisat
  __builtin_HEXAGON_V6_vdmpyhsat_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat.128B
  __builtin_HEXAGON_V6_vdmpyhsat_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhsat_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat.acc
  __builtin_HEXAGON_V6_vdmpyhsat(v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsat
  __builtin_HEXAGON_V6_vdmpyhsuisat_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat.128B
  __builtin_HEXAGON_V6_vdmpyhsuisat_acc_128B(v32, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhsuisat_acc(v16, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat.acc
  __builtin_HEXAGON_V6_vdmpyhsuisat(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsuisat
  __builtin_HEXAGON_V6_vdmpyhsusat_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat.128B
  __builtin_HEXAGON_V6_vdmpyhsusat_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhsusat_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat.acc
  __builtin_HEXAGON_V6_vdmpyhsusat(v16, 0);
  // CHECK: @llvm.hexagon.V6.vdmpyhsusat
  __builtin_HEXAGON_V6_vdmpyhvsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat.128B
  __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat.acc.128B
  __builtin_HEXAGON_V6_vdmpyhvsat_acc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat.acc
  __builtin_HEXAGON_V6_vdmpyhvsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vdmpyhvsat
  __builtin_HEXAGON_V6_vdsaduh_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vdsaduh.128B
  __builtin_HEXAGON_V6_vdsaduh_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vdsaduh.acc.128B
  __builtin_HEXAGON_V6_vdsaduh_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vdsaduh.acc
  __builtin_HEXAGON_V6_vdsaduh(v32, 0);
  // CHECK: @llvm.hexagon.V6.vdsaduh
  __builtin_HEXAGON_V6_veqb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.veqb.128B
  __builtin_HEXAGON_V6_veqb_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqb.and.128B
  __builtin_HEXAGON_V6_veqb_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqb.and
  __builtin_HEXAGON_V6_veqb_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqb.or.128B
  __builtin_HEXAGON_V6_veqb_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqb.or
  __builtin_HEXAGON_V6_veqb(v16, v16);
  // CHECK: @llvm.hexagon.V6.veqb
  __builtin_HEXAGON_V6_veqb_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqb.xor.128B
  __builtin_HEXAGON_V6_veqb_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqb.xor
  __builtin_HEXAGON_V6_veqh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.veqh.128B
  __builtin_HEXAGON_V6_veqh_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqh.and.128B
  __builtin_HEXAGON_V6_veqh_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqh.and
  __builtin_HEXAGON_V6_veqh_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqh.or.128B
  __builtin_HEXAGON_V6_veqh_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqh.or
  __builtin_HEXAGON_V6_veqh(v16, v16);
  // CHECK: @llvm.hexagon.V6.veqh
  __builtin_HEXAGON_V6_veqh_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqh.xor.128B
  __builtin_HEXAGON_V6_veqh_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqh.xor
  __builtin_HEXAGON_V6_veqw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.veqw.128B
  __builtin_HEXAGON_V6_veqw_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqw.and.128B
  __builtin_HEXAGON_V6_veqw_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqw.and
  __builtin_HEXAGON_V6_veqw_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqw.or.128B
  __builtin_HEXAGON_V6_veqw_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqw.or
  __builtin_HEXAGON_V6_veqw(v16, v16);
  // CHECK: @llvm.hexagon.V6.veqw
  __builtin_HEXAGON_V6_veqw_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.veqw.xor.128B
  __builtin_HEXAGON_V6_veqw_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.veqw.xor
  __builtin_HEXAGON_V6_vgtb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtb.128B
  __builtin_HEXAGON_V6_vgtb_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtb.and.128B
  __builtin_HEXAGON_V6_vgtb_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtb.and
  __builtin_HEXAGON_V6_vgtb_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtb.or.128B
  __builtin_HEXAGON_V6_vgtb_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtb.or
  __builtin_HEXAGON_V6_vgtb(v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtb
  __builtin_HEXAGON_V6_vgtb_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtb.xor.128B
  __builtin_HEXAGON_V6_vgtb_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtb.xor
  __builtin_HEXAGON_V6_vgth_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vgth.128B
  __builtin_HEXAGON_V6_vgth_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgth.and.128B
  __builtin_HEXAGON_V6_vgth_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgth.and
  __builtin_HEXAGON_V6_vgth_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgth.or.128B
  __builtin_HEXAGON_V6_vgth_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgth.or
  __builtin_HEXAGON_V6_vgth(v16, v16);
  // CHECK: @llvm.hexagon.V6.vgth
  __builtin_HEXAGON_V6_vgth_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgth.xor.128B
  __builtin_HEXAGON_V6_vgth_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgth.xor
  __builtin_HEXAGON_V6_vgtub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtub.128B
  __builtin_HEXAGON_V6_vgtub_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtub.and.128B
  __builtin_HEXAGON_V6_vgtub_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtub.and
  __builtin_HEXAGON_V6_vgtub_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtub.or.128B
  __builtin_HEXAGON_V6_vgtub_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtub.or
  __builtin_HEXAGON_V6_vgtub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtub
  __builtin_HEXAGON_V6_vgtub_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtub.xor.128B
  __builtin_HEXAGON_V6_vgtub_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtub.xor
  __builtin_HEXAGON_V6_vgtuh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuh.128B
  __builtin_HEXAGON_V6_vgtuh_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuh.and.128B
  __builtin_HEXAGON_V6_vgtuh_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuh.and
  __builtin_HEXAGON_V6_vgtuh_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuh.or.128B
  __builtin_HEXAGON_V6_vgtuh_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuh.or
  __builtin_HEXAGON_V6_vgtuh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuh
  __builtin_HEXAGON_V6_vgtuh_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuh.xor.128B
  __builtin_HEXAGON_V6_vgtuh_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuh.xor
  __builtin_HEXAGON_V6_vgtuw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuw.128B
  __builtin_HEXAGON_V6_vgtuw_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuw.and.128B
  __builtin_HEXAGON_V6_vgtuw_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuw.and
  __builtin_HEXAGON_V6_vgtuw_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuw.or.128B
  __builtin_HEXAGON_V6_vgtuw_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuw.or
  __builtin_HEXAGON_V6_vgtuw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuw
  __builtin_HEXAGON_V6_vgtuw_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtuw.xor.128B
  __builtin_HEXAGON_V6_vgtuw_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtuw.xor
  __builtin_HEXAGON_V6_vgtw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtw.128B
  __builtin_HEXAGON_V6_vgtw_and_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtw.and.128B
  __builtin_HEXAGON_V6_vgtw_and(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtw.and
  __builtin_HEXAGON_V6_vgtw_or_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtw.or.128B
  __builtin_HEXAGON_V6_vgtw_or(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtw.or
  __builtin_HEXAGON_V6_vgtw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtw
  __builtin_HEXAGON_V6_vgtw_xor_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vgtw.xor.128B
  __builtin_HEXAGON_V6_vgtw_xor(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vgtw.xor
  __builtin_HEXAGON_V6_vinsertwr_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vinsertwr.128B
  __builtin_HEXAGON_V6_vinsertwr(v16, 0);
  // CHECK: @llvm.hexagon.V6.vinsertwr
  __builtin_HEXAGON_V6_vlalignb_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vlalignb.128B
  __builtin_HEXAGON_V6_vlalignbi_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vlalignbi.128B
  __builtin_HEXAGON_V6_vlalignbi(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vlalignbi
  __builtin_HEXAGON_V6_vlalignb(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vlalignb
  __builtin_HEXAGON_V6_vlsrh_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vlsrh.128B
  __builtin_HEXAGON_V6_vlsrhv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vlsrhv.128B
  __builtin_HEXAGON_V6_vlsrh(v16, 0);
  // CHECK: @llvm.hexagon.V6.vlsrh
  __builtin_HEXAGON_V6_vlsrhv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vlsrhv
  __builtin_HEXAGON_V6_vlsrw_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vlsrw.128B
  __builtin_HEXAGON_V6_vlsrwv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vlsrwv.128B
  __builtin_HEXAGON_V6_vlsrw(v16, 0);
  // CHECK: @llvm.hexagon.V6.vlsrw
  __builtin_HEXAGON_V6_vlsrwv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vlsrwv
  __builtin_HEXAGON_V6_vlutvvb_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.128B
  __builtin_HEXAGON_V6_vlutvvb_oracc_128B(v32, v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.oracc.128B
  __builtin_HEXAGON_V6_vlutvvb_oracc(v16, v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb.oracc
  __builtin_HEXAGON_V6_vlutvvb(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vlutvvb
  __builtin_HEXAGON_V6_vlutvwh_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.128B
  __builtin_HEXAGON_V6_vlutvwh_oracc_128B(v64, v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.oracc.128B
  __builtin_HEXAGON_V6_vlutvwh_oracc(v32, v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh.oracc
  __builtin_HEXAGON_V6_vlutvwh(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vlutvwh
  __builtin_HEXAGON_V6_vmaxh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmaxh.128B
  __builtin_HEXAGON_V6_vmaxh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmaxh
  __builtin_HEXAGON_V6_vmaxub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmaxub.128B
  __builtin_HEXAGON_V6_vmaxub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmaxub
  __builtin_HEXAGON_V6_vmaxuh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmaxuh.128B
  __builtin_HEXAGON_V6_vmaxuh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmaxuh
  __builtin_HEXAGON_V6_vmaxw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmaxw.128B
  __builtin_HEXAGON_V6_vmaxw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmaxw
  __builtin_HEXAGON_V6_vminh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vminh.128B
  __builtin_HEXAGON_V6_vminh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vminh
  __builtin_HEXAGON_V6_vminub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vminub.128B
  __builtin_HEXAGON_V6_vminub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vminub
  __builtin_HEXAGON_V6_vminuh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vminuh.128B
  __builtin_HEXAGON_V6_vminuh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vminuh
  __builtin_HEXAGON_V6_vminw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vminw.128B
  __builtin_HEXAGON_V6_vminw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vminw
  __builtin_HEXAGON_V6_vmpabus_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpabus.128B
  __builtin_HEXAGON_V6_vmpabus_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpabus.acc.128B
  __builtin_HEXAGON_V6_vmpabus_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpabus.acc
  __builtin_HEXAGON_V6_vmpabusv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpabusv.128B
  __builtin_HEXAGON_V6_vmpabus(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpabus
  __builtin_HEXAGON_V6_vmpabusv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpabusv
  __builtin_HEXAGON_V6_vmpabuuv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vmpabuuv.128B
  __builtin_HEXAGON_V6_vmpabuuv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpabuuv
  __builtin_HEXAGON_V6_vmpahb_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpahb.128B
  __builtin_HEXAGON_V6_vmpahb_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vmpahb.acc.128B
  __builtin_HEXAGON_V6_vmpahb_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpahb.acc
  __builtin_HEXAGON_V6_vmpahb(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpahb
  __builtin_HEXAGON_V6_vmpybus_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus.128B
  __builtin_HEXAGON_V6_vmpybus_acc_128B(v64, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus.acc.128B
  __builtin_HEXAGON_V6_vmpybus_acc(v32, v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus.acc
  __builtin_HEXAGON_V6_vmpybusv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpybusv.128B
  __builtin_HEXAGON_V6_vmpybus(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpybus
  __builtin_HEXAGON_V6_vmpybusv_acc_128B(v64, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpybusv.acc.128B
  __builtin_HEXAGON_V6_vmpybusv_acc(v32, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpybusv.acc
  __builtin_HEXAGON_V6_vmpybusv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpybusv
  __builtin_HEXAGON_V6_vmpybv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpybv.128B
  __builtin_HEXAGON_V6_vmpybv_acc_128B(v64, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpybv.acc.128B
  __builtin_HEXAGON_V6_vmpybv_acc(v32, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpybv.acc
  __builtin_HEXAGON_V6_vmpybv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpybv
  __builtin_HEXAGON_V6_vmpyewuh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyewuh.128B
  __builtin_HEXAGON_V6_vmpyewuh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyewuh
  __builtin_HEXAGON_V6_vmpyh_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyh.128B
  __builtin_HEXAGON_V6_vmpyhsat_acc_128B(v64, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsat.acc.128B
  __builtin_HEXAGON_V6_vmpyhsat_acc(v32, v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsat.acc
  __builtin_HEXAGON_V6_vmpyhsrs_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsrs.128B
  __builtin_HEXAGON_V6_vmpyhsrs(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhsrs
  __builtin_HEXAGON_V6_vmpyhss_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhss.128B
  __builtin_HEXAGON_V6_vmpyhss(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyhss
  __builtin_HEXAGON_V6_vmpyhus_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyhus.128B
  __builtin_HEXAGON_V6_vmpyhus_acc_128B(v64, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyhus.acc.128B
  __builtin_HEXAGON_V6_vmpyhus_acc(v32, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyhus.acc
  __builtin_HEXAGON_V6_vmpyhus(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyhus
  __builtin_HEXAGON_V6_vmpyhv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyhv.128B
  __builtin_HEXAGON_V6_vmpyh(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyh
  __builtin_HEXAGON_V6_vmpyhv_acc_128B(v64, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyhv.acc.128B
  __builtin_HEXAGON_V6_vmpyhv_acc(v32, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyhv.acc
  __builtin_HEXAGON_V6_vmpyhvsrs_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyhvsrs.128B
  __builtin_HEXAGON_V6_vmpyhvsrs(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyhvsrs
  __builtin_HEXAGON_V6_vmpyhv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyhv
  __builtin_HEXAGON_V6_vmpyieoh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyieoh.128B
  __builtin_HEXAGON_V6_vmpyieoh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyieoh
  __builtin_HEXAGON_V6_vmpyiewh_acc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyiewh.acc.128B
  __builtin_HEXAGON_V6_vmpyiewh_acc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyiewh.acc
  __builtin_HEXAGON_V6_vmpyiewuh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh.128B
  __builtin_HEXAGON_V6_vmpyiewuh_acc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh.acc.128B
  __builtin_HEXAGON_V6_vmpyiewuh_acc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh.acc
  __builtin_HEXAGON_V6_vmpyiewuh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyiewuh
  __builtin_HEXAGON_V6_vmpyih_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyih.128B
  __builtin_HEXAGON_V6_vmpyih_acc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyih.acc.128B
  __builtin_HEXAGON_V6_vmpyih_acc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyih.acc
  __builtin_HEXAGON_V6_vmpyihb_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyihb.128B
  __builtin_HEXAGON_V6_vmpyihb_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyihb.acc.128B
  __builtin_HEXAGON_V6_vmpyihb_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyihb.acc
  __builtin_HEXAGON_V6_vmpyihb(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyihb
  __builtin_HEXAGON_V6_vmpyih(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyih
  __builtin_HEXAGON_V6_vmpyiowh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyiowh.128B
  __builtin_HEXAGON_V6_vmpyiowh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyiowh
  __builtin_HEXAGON_V6_vmpyiwb_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwb.128B
  __builtin_HEXAGON_V6_vmpyiwb_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwb.acc.128B
  __builtin_HEXAGON_V6_vmpyiwb_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwb.acc
  __builtin_HEXAGON_V6_vmpyiwb(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwb
  __builtin_HEXAGON_V6_vmpyiwh_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh.128B
  __builtin_HEXAGON_V6_vmpyiwh_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh.acc.128B
  __builtin_HEXAGON_V6_vmpyiwh_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh.acc
  __builtin_HEXAGON_V6_vmpyiwh(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyiwh
  __builtin_HEXAGON_V6_vmpyowh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyowh.128B
  __builtin_HEXAGON_V6_vmpyowh_rnd_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd.128B
  __builtin_HEXAGON_V6_vmpyowh_rnd_sacc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd.sacc.128B
  __builtin_HEXAGON_V6_vmpyowh_rnd_sacc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd.sacc
  __builtin_HEXAGON_V6_vmpyowh_rnd(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyowh.rnd
  __builtin_HEXAGON_V6_vmpyowh_sacc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyowh.sacc.128B
  __builtin_HEXAGON_V6_vmpyowh_sacc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyowh.sacc
  __builtin_HEXAGON_V6_vmpyowh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyowh
  __builtin_HEXAGON_V6_vmpyub_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyub.128B
  __builtin_HEXAGON_V6_vmpyub_acc_128B(v64, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyub.acc.128B
  __builtin_HEXAGON_V6_vmpyub_acc(v32, v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyub.acc
  __builtin_HEXAGON_V6_vmpyubv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyubv.128B
  __builtin_HEXAGON_V6_vmpyub(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyub
  __builtin_HEXAGON_V6_vmpyubv_acc_128B(v64, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyubv.acc.128B
  __builtin_HEXAGON_V6_vmpyubv_acc(v32, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyubv.acc
  __builtin_HEXAGON_V6_vmpyubv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyubv
  __builtin_HEXAGON_V6_vmpyuh_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuh.128B
  __builtin_HEXAGON_V6_vmpyuh_acc_128B(v64, v32, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuh.acc.128B
  __builtin_HEXAGON_V6_vmpyuh_acc(v32, v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuh.acc
  __builtin_HEXAGON_V6_vmpyuhv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyuhv.128B
  __builtin_HEXAGON_V6_vmpyuh(v16, 0);
  // CHECK: @llvm.hexagon.V6.vmpyuh
  __builtin_HEXAGON_V6_vmpyuhv_acc_128B(v64, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmpyuhv.acc.128B
  __builtin_HEXAGON_V6_vmpyuhv_acc(v32, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyuhv.acc
  __builtin_HEXAGON_V6_vmpyuhv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vmpyuhv
  __builtin_HEXAGON_V6_vmux_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vmux.128B
  __builtin_HEXAGON_V6_vmux(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vmux
  __builtin_HEXAGON_V6_vnavgh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vnavgh.128B
  __builtin_HEXAGON_V6_vnavgh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vnavgh
  __builtin_HEXAGON_V6_vnavgub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vnavgub.128B
  __builtin_HEXAGON_V6_vnavgub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vnavgub
  __builtin_HEXAGON_V6_vnavgw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vnavgw.128B
  __builtin_HEXAGON_V6_vnavgw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vnavgw
  __builtin_HEXAGON_V6_vnormamth_128B(v32);
  // CHECK: @llvm.hexagon.V6.vnormamth.128B
  __builtin_HEXAGON_V6_vnormamth(v16);
  // CHECK: @llvm.hexagon.V6.vnormamth
  __builtin_HEXAGON_V6_vnormamtw_128B(v32);
  // CHECK: @llvm.hexagon.V6.vnormamtw.128B
  __builtin_HEXAGON_V6_vnormamtw(v16);
  // CHECK: @llvm.hexagon.V6.vnormamtw
  __builtin_HEXAGON_V6_vnot_128B(v32);
  // CHECK: @llvm.hexagon.V6.vnot.128B
  __builtin_HEXAGON_V6_vnot(v16);
  // CHECK: @llvm.hexagon.V6.vnot
  __builtin_HEXAGON_V6_vor_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vor.128B
  __builtin_HEXAGON_V6_vor(v16, v16);
  // CHECK: @llvm.hexagon.V6.vor
  __builtin_HEXAGON_V6_vpackeb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackeb.128B
  __builtin_HEXAGON_V6_vpackeb(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackeb
  __builtin_HEXAGON_V6_vpackeh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackeh.128B
  __builtin_HEXAGON_V6_vpackeh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackeh
  __builtin_HEXAGON_V6_vpackhb_sat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackhb.sat.128B
  __builtin_HEXAGON_V6_vpackhb_sat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackhb.sat
  __builtin_HEXAGON_V6_vpackhub_sat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackhub.sat.128B
  __builtin_HEXAGON_V6_vpackhub_sat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackhub.sat
  __builtin_HEXAGON_V6_vpackob_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackob.128B
  __builtin_HEXAGON_V6_vpackob(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackob
  __builtin_HEXAGON_V6_vpackoh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackoh.128B
  __builtin_HEXAGON_V6_vpackoh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackoh
  __builtin_HEXAGON_V6_vpackwh_sat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackwh.sat.128B
  __builtin_HEXAGON_V6_vpackwh_sat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackwh.sat
  __builtin_HEXAGON_V6_vpackwuh_sat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vpackwuh.sat.128B
  __builtin_HEXAGON_V6_vpackwuh_sat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vpackwuh.sat
  __builtin_HEXAGON_V6_vpopcounth_128B(v32);
  // CHECK: @llvm.hexagon.V6.vpopcounth.128B
  __builtin_HEXAGON_V6_vpopcounth(v16);
  // CHECK: @llvm.hexagon.V6.vpopcounth
  __builtin_HEXAGON_V6_vrdelta_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vrdelta.128B
  __builtin_HEXAGON_V6_vrdelta(v16, v16);
  // CHECK: @llvm.hexagon.V6.vrdelta
  __builtin_HEXAGON_V6_vrmpybus_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus.128B
  __builtin_HEXAGON_V6_vrmpybus_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus.acc.128B
  __builtin_HEXAGON_V6_vrmpybus_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus.acc
  __builtin_HEXAGON_V6_vrmpybusi_128B(v64, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi.128B
  __builtin_HEXAGON_V6_vrmpybusi_acc_128B(v64, v64, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi.acc.128B
  __builtin_HEXAGON_V6_vrmpybusi_acc(v32, v32, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi.acc
  __builtin_HEXAGON_V6_vrmpybusi(v32, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybusi
  __builtin_HEXAGON_V6_vrmpybusv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vrmpybusv.128B
  __builtin_HEXAGON_V6_vrmpybus(v16, 0);
  // CHECK: @llvm.hexagon.V6.vrmpybus
  __builtin_HEXAGON_V6_vrmpybusv_acc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vrmpybusv.acc.128B
  __builtin_HEXAGON_V6_vrmpybusv_acc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vrmpybusv.acc
  __builtin_HEXAGON_V6_vrmpybusv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vrmpybusv
  __builtin_HEXAGON_V6_vrmpybv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vrmpybv.128B
  __builtin_HEXAGON_V6_vrmpybv_acc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vrmpybv.acc.128B
  __builtin_HEXAGON_V6_vrmpybv_acc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vrmpybv.acc
  __builtin_HEXAGON_V6_vrmpybv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vrmpybv
  __builtin_HEXAGON_V6_vrmpyub_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.128B
  __builtin_HEXAGON_V6_vrmpyub_acc_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.acc.128B
  __builtin_HEXAGON_V6_vrmpyub_acc(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub.acc
  __builtin_HEXAGON_V6_vrmpyubi_128B(v64, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi.128B
  __builtin_HEXAGON_V6_vrmpyubi_acc_128B(v64, v64, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi.acc.128B
  __builtin_HEXAGON_V6_vrmpyubi_acc(v32, v32, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi.acc
  __builtin_HEXAGON_V6_vrmpyubi(v32, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyubi
  __builtin_HEXAGON_V6_vrmpyubv_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vrmpyubv.128B
  __builtin_HEXAGON_V6_vrmpyub(v16, 0);
  // CHECK: @llvm.hexagon.V6.vrmpyub
  __builtin_HEXAGON_V6_vrmpyubv_acc_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vrmpyubv.acc.128B
  __builtin_HEXAGON_V6_vrmpyubv_acc(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vrmpyubv.acc
  __builtin_HEXAGON_V6_vrmpyubv(v16, v16);
  // CHECK: @llvm.hexagon.V6.vrmpyubv
  __builtin_HEXAGON_V6_vror_128B(v32, 0);
  // CHECK: @llvm.hexagon.V6.vror.128B
  __builtin_HEXAGON_V6_vror(v16, 0);
  // CHECK: @llvm.hexagon.V6.vror
  __builtin_HEXAGON_V6_vroundhb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vroundhb.128B
  __builtin_HEXAGON_V6_vroundhb(v16, v16);
  // CHECK: @llvm.hexagon.V6.vroundhb
  __builtin_HEXAGON_V6_vroundhub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vroundhub.128B
  __builtin_HEXAGON_V6_vroundhub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vroundhub
  __builtin_HEXAGON_V6_vroundwh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vroundwh.128B
  __builtin_HEXAGON_V6_vroundwh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vroundwh
  __builtin_HEXAGON_V6_vroundwuh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vroundwuh.128B
  __builtin_HEXAGON_V6_vroundwuh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vroundwuh
  __builtin_HEXAGON_V6_vrsadubi_128B(v64, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrsadubi.128B
  __builtin_HEXAGON_V6_vrsadubi_acc_128B(v64, v64, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrsadubi.acc.128B
  __builtin_HEXAGON_V6_vrsadubi_acc(v32, v32, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrsadubi.acc
  __builtin_HEXAGON_V6_vrsadubi(v32, 0, 0);
  // CHECK: @llvm.hexagon.V6.vrsadubi
  __builtin_HEXAGON_V6_vsathub_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsathub.128B
  __builtin_HEXAGON_V6_vsathub(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsathub
  __builtin_HEXAGON_V6_vsatwh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsatwh.128B
  __builtin_HEXAGON_V6_vsatwh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsatwh
  __builtin_HEXAGON_V6_vsb_128B(v32);
  // CHECK: @llvm.hexagon.V6.vsb.128B
  __builtin_HEXAGON_V6_vsb(v16);
  // CHECK: @llvm.hexagon.V6.vsb
  __builtin_HEXAGON_V6_vsh_128B(v32);
  // CHECK: @llvm.hexagon.V6.vsh.128B
  __builtin_HEXAGON_V6_vshufeh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vshufeh.128B
  __builtin_HEXAGON_V6_vshufeh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vshufeh
  __builtin_HEXAGON_V6_vshuffb_128B(v32);
  // CHECK: @llvm.hexagon.V6.vshuffb.128B
  __builtin_HEXAGON_V6_vshuffb(v16);
  // CHECK: @llvm.hexagon.V6.vshuffb
  __builtin_HEXAGON_V6_vshuffeb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vshuffeb.128B
  __builtin_HEXAGON_V6_vshuffeb(v16, v16);
  // CHECK: @llvm.hexagon.V6.vshuffeb
  __builtin_HEXAGON_V6_vshuffh_128B(v32);
  // CHECK: @llvm.hexagon.V6.vshuffh.128B
  __builtin_HEXAGON_V6_vshuffh(v16);
  // CHECK: @llvm.hexagon.V6.vshuffh
  __builtin_HEXAGON_V6_vshuffob_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vshuffob.128B
  __builtin_HEXAGON_V6_vshuffob(v16, v16);
  // CHECK: @llvm.hexagon.V6.vshuffob
  __builtin_HEXAGON_V6_vshuffvdd_128B(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vshuffvdd.128B
  __builtin_HEXAGON_V6_vshuffvdd(v16, v16, 0);
  // CHECK: @llvm.hexagon.V6.vshuffvdd
  __builtin_HEXAGON_V6_vshufoeb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vshufoeb.128B
  __builtin_HEXAGON_V6_vshufoeb(v16, v16);
  // CHECK: @llvm.hexagon.V6.vshufoeb
  __builtin_HEXAGON_V6_vshufoeh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vshufoeh.128B
  __builtin_HEXAGON_V6_vshufoeh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vshufoeh
  __builtin_HEXAGON_V6_vshufoh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vshufoh.128B
  __builtin_HEXAGON_V6_vshufoh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vshufoh
  __builtin_HEXAGON_V6_vsh(v16);
  // CHECK: @llvm.hexagon.V6.vsh
  __builtin_HEXAGON_V6_vsubb_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubb.128B
  __builtin_HEXAGON_V6_vsubb_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubb.dv.128B
  __builtin_HEXAGON_V6_vsubb_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubb.dv
  __builtin_HEXAGON_V6_vsubbnq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubbnq.128B
  __builtin_HEXAGON_V6_vsubbnq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubbnq
  __builtin_HEXAGON_V6_vsubbq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubbq.128B
  __builtin_HEXAGON_V6_vsubbq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubbq
  __builtin_HEXAGON_V6_vsubb(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubb
  __builtin_HEXAGON_V6_vsubh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubh.128B
  __builtin_HEXAGON_V6_vsubh_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubh.dv.128B
  __builtin_HEXAGON_V6_vsubh_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubh.dv
  __builtin_HEXAGON_V6_vsubhnq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubhnq.128B
  __builtin_HEXAGON_V6_vsubhnq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubhnq
  __builtin_HEXAGON_V6_vsubhq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubhq.128B
  __builtin_HEXAGON_V6_vsubhq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubhq
  __builtin_HEXAGON_V6_vsubhsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubhsat.128B
  __builtin_HEXAGON_V6_vsubhsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubhsat.dv.128B
  __builtin_HEXAGON_V6_vsubhsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubhsat.dv
  __builtin_HEXAGON_V6_vsubhsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubhsat
  __builtin_HEXAGON_V6_vsubh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubh
  __builtin_HEXAGON_V6_vsubhw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubhw.128B
  __builtin_HEXAGON_V6_vsubhw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubhw
  __builtin_HEXAGON_V6_vsububh_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsububh.128B
  __builtin_HEXAGON_V6_vsububh(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsububh
  __builtin_HEXAGON_V6_vsububsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsububsat.128B
  __builtin_HEXAGON_V6_vsububsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsububsat.dv.128B
  __builtin_HEXAGON_V6_vsububsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsububsat.dv
  __builtin_HEXAGON_V6_vsububsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsububsat
  __builtin_HEXAGON_V6_vsubuhsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubuhsat.128B
  __builtin_HEXAGON_V6_vsubuhsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubuhsat.dv.128B
  __builtin_HEXAGON_V6_vsubuhsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubuhsat.dv
  __builtin_HEXAGON_V6_vsubuhsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubuhsat
  __builtin_HEXAGON_V6_vsubuhw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubuhw.128B
  __builtin_HEXAGON_V6_vsubuhw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubuhw
  __builtin_HEXAGON_V6_vsubw_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubw.128B
  __builtin_HEXAGON_V6_vsubw_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubw.dv.128B
  __builtin_HEXAGON_V6_vsubw_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubw.dv
  __builtin_HEXAGON_V6_vsubwnq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubwnq.128B
  __builtin_HEXAGON_V6_vsubwnq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubwnq
  __builtin_HEXAGON_V6_vsubwq_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubwq.128B
  __builtin_HEXAGON_V6_vsubwq(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubwq
  __builtin_HEXAGON_V6_vsubwsat_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubwsat.128B
  __builtin_HEXAGON_V6_vsubwsat_dv_128B(v64, v64);
  // CHECK: @llvm.hexagon.V6.vsubwsat.dv.128B
  __builtin_HEXAGON_V6_vsubwsat_dv(v32, v32);
  // CHECK: @llvm.hexagon.V6.vsubwsat.dv
  __builtin_HEXAGON_V6_vsubwsat(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubwsat
  __builtin_HEXAGON_V6_vsubw(v16, v16);
  // CHECK: @llvm.hexagon.V6.vsubw
  __builtin_HEXAGON_V6_vswap_128B(v32, v32, v32);
  // CHECK: @llvm.hexagon.V6.vswap.128B
  __builtin_HEXAGON_V6_vswap(v16, v16, v16);
  // CHECK: @llvm.hexagon.V6.vswap
  __builtin_HEXAGON_V6_vtmpyb_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyb.128B
  __builtin_HEXAGON_V6_vtmpyb_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyb.acc.128B
  __builtin_HEXAGON_V6_vtmpyb_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyb.acc
  __builtin_HEXAGON_V6_vtmpybus_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus.128B
  __builtin_HEXAGON_V6_vtmpybus_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus.acc.128B
  __builtin_HEXAGON_V6_vtmpybus_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus.acc
  __builtin_HEXAGON_V6_vtmpybus(v32, 0);
  // CHECK: @llvm.hexagon.V6.vtmpybus
  __builtin_HEXAGON_V6_vtmpyb(v32, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyb
  __builtin_HEXAGON_V6_vtmpyhb_128B(v64, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb.128B
  __builtin_HEXAGON_V6_vtmpyhb_acc_128B(v64, v64, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb.acc.128B
  __builtin_HEXAGON_V6_vtmpyhb_acc(v32, v32, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb.acc
  __builtin_HEXAGON_V6_vtmpyhb(v32, 0);
  // CHECK: @llvm.hexagon.V6.vtmpyhb
  __builtin_HEXAGON_V6_vunpackb_128B(v32);
  // CHECK: @llvm.hexagon.V6.vunpackb.128B
  __builtin_HEXAGON_V6_vunpackb(v16);
  // CHECK: @llvm.hexagon.V6.vunpackb
  __builtin_HEXAGON_V6_vunpackh_128B(v32);
  // CHECK: @llvm.hexagon.V6.vunpackh.128B
  __builtin_HEXAGON_V6_vunpackh(v16);
  // CHECK: @llvm.hexagon.V6.vunpackh
  __builtin_HEXAGON_V6_vunpackob_128B(v64, v32);
  // CHECK: @llvm.hexagon.V6.vunpackob.128B
  __builtin_HEXAGON_V6_vunpackob(v32, v16);
  // CHECK: @llvm.hexagon.V6.vunpackob
  __builtin_HEXAGON_V6_vunpackoh_128B(v64, v32);
  // CHECK: @llvm.hexagon.V6.vunpackoh.128B
  __builtin_HEXAGON_V6_vunpackoh(v32, v16);
  // CHECK: @llvm.hexagon.V6.vunpackoh
  __builtin_HEXAGON_V6_vunpackub_128B(v32);
  // CHECK: @llvm.hexagon.V6.vunpackub.128B
  __builtin_HEXAGON_V6_vunpackub(v16);
  // CHECK: @llvm.hexagon.V6.vunpackub
  __builtin_HEXAGON_V6_vunpackuh_128B(v32);
  // CHECK: @llvm.hexagon.V6.vunpackuh.128B
  __builtin_HEXAGON_V6_vunpackuh(v16);
  // CHECK: @llvm.hexagon.V6.vunpackuh
  __builtin_HEXAGON_V6_vxor_128B(v32, v32);
  // CHECK: @llvm.hexagon.V6.vxor.128B
  __builtin_HEXAGON_V6_vxor(v16, v16);
  // CHECK: @llvm.hexagon.V6.vxor
  __builtin_HEXAGON_V6_vzb_128B(v32);
  // CHECK: @llvm.hexagon.V6.vzb.128B
  __builtin_HEXAGON_V6_vzb(v16);
  // CHECK: @llvm.hexagon.V6.vzb
  __builtin_HEXAGON_V6_vzh_128B(v32);
  // CHECK: @llvm.hexagon.V6.vzh.128B
  __builtin_HEXAGON_V6_vzh(v16);
  // CHECK: @llvm.hexagon.V6.vzh
  __builtin_HEXAGON_Y2_dccleana(0);
  // CHECK: @llvm.hexagon.Y2.dccleana
  __builtin_HEXAGON_Y2_dccleaninva(0);
  // CHECK: @llvm.hexagon.Y2.dccleaninva
  __builtin_HEXAGON_Y2_dcinva(0);
  // CHECK: @llvm.hexagon.Y2.dcinva
  __builtin_HEXAGON_Y2_dczeroa(0);
  // CHECK: @llvm.hexagon.Y2.dczeroa
  __builtin_HEXAGON_Y4_l2fetch(0, 0);
  // CHECK: @llvm.hexagon.Y4.l2fetch
  __builtin_HEXAGON_Y5_l2fetch(0, 0);
  // CHECK: @llvm.hexagon.Y5.l2fetch
}
