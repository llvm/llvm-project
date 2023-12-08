// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +altivec \
// RUN: -target-feature +power8-vector -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include <altivec.h>
vector signed long long vsll1, vsll2, vsll3;
vector unsigned long long vull1, vull2, vull3;
vector signed char vsc;
vector unsigned char vuc;
vector bool long long vbll;

void dummy();
void test() {
  vec_abs(vsll1);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vmaxsd
  dummy();
// CHECK-NEXT: call void @dummy()
  vec_add(vsll1, vsll2);
// CHECK: add <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_and(vsll1, vsll2);
// CHECK: and <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_vand(vsll1, vsll2);
// CHECK: and <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_andc(vsll1, vsll2);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_vandc(vsll1, vsll2);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_cmpeq(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpequd
  dummy();
// CHECK: call void @dummy()
  vec_cmpne(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpequd
// CHECK: xor <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_cmpgt(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpgtsd
  dummy();
// CHECK: call void @dummy()
  vec_cmpge(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpgtsd
// CHECK: xor <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_cmple(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpgtsd
// CHECK: xor <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_cmplt(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpgtsd
  dummy();
// CHECK: call void @dummy()
  vec_popcnt(vsll1);
// CHECK: call <2 x i64> @llvm.ctpop.v2i64
  dummy();
// CHECK: call void @dummy()
  vec_cntlz(vsll1);
// CHECK: call <2 x i64> @llvm.ctlz.v2i64
  dummy();
// CHECK: call void @dummy()
  vec_float2(vsll1, vsll2);
// CHECK: sitofp i64 %{{.*}} to float
  dummy();
// CHECK: call void @dummy()
  vec_floate(vsll1);
// CHECK: call <4 x float> @llvm.ppc.vsx.xvcvsxdsp
  dummy();
// CHECK: call void @dummy()
  vec_floato(vsll1);
// CHECK: call <4 x float> @llvm.ppc.vsx.xvcvsxdsp
  dummy();
// CHECK: call void @dummy()
  vec_double(vsll1);
// CHECK: sitofp <2 x i64> %{{.*}} to <2 x double>
  dummy();
// CHECK: call void @dummy()
  vec_div(vsll1, vsll2);
// CHECK: sdiv <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_eqv(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xxleqv
  dummy();
// CHECK: call void @dummy()
  vec_max(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vmaxsd
  dummy();
// CHECK: call void @dummy()
  vec_mergeh(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_mergel(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_mergee(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_mergeo(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_min(vsll1, vsll2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vminsd
  dummy();
// CHECK: call void @dummy()
  vec_mul(vsll1, vsll2);
// CHECK: mul <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_nand(vsll1, vsll2);
// CHECK: and <2 x i64>
// CHECK: xor <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_nor(vsll1, vsll2);
// CHECK: or <2 x i64>
// CHECK: xor <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_or(vsll1, vsll2);
// CHECK: or <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_orc(vsll1, vsll2);
// CHECK: xor <2 x i64>
// CHECK: or <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_vor(vsll1, vsll2);
// CHECK: or <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_pack(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_vpkudum(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_packs(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vpksdss
  dummy();
// CHECK: call void @dummy()
  vec_vpkudus(vull1, vull2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vpkudus
  dummy();
// CHECK: call void @dummy()
  vec_packsu(vsll1, vsll2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vpksdus
  dummy();
// CHECK: call void @dummy()
  vec_rl(vull1, vull2);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vrld
  dummy();
// CHECK: call void @dummy()
  vec_sel(vsll1, vsll2, vbll);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK: and <2 x i64>
// CHECK: or <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_sl(vull1, vull2);
// CHECK: shl <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_sld(vsll1, vsll2, 2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_sldw(vsll1, vsll2, 2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_sll(vsll1, vuc);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vsl
  dummy();
// CHECK: call void @dummy()
  vec_slo(vsll1, vsc);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vslo
  dummy();
// CHECK: call void @dummy()
  vec_splat(vsll1, 2);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
  dummy();
// CHECK: call void @dummy()
  vec_sr(vull1, vull2);
// CHECK: lshr <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_sra(vsll1, vull2);
// CHECK: ashr <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_srl(vsll1, vuc);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vsr
  dummy();
// CHECK: call void @dummy()
  vec_sro(vsll1, vsc);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vsro
  dummy();
// CHECK: call void @dummy()
  vec_sub(vsll1, vsll2);
// CHECK: sub <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_xor(vsll1, vsll2);
// CHECK: xor <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_vxor(vsll1, vsll2);
// CHECK: xor <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_extract(vsll1, 2);
// CHECK: extractelement <2 x i64>
  dummy();
// CHECK: call void @dummy()
  vec_all_eq(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpequd.p
  dummy();
// CHECK: call void @dummy()
  vec_all_ge(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_all_gt(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_all_le(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_all_lt(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_all_ne(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpequd.p
  dummy();
// CHECK: call void @dummy()
  vec_any_eq(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpequd.p
  dummy();
// CHECK: call void @dummy()
  vec_any_ge(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_any_gt(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_any_le(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_any_lt(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpgtsd.p
  dummy();
// CHECK: call void @dummy()
  vec_any_ne(vsll1, vsll2);
// CHECK: call i32 @llvm.ppc.altivec.vcmpequd.p
  dummy();
// CHECK: call void @dummy()
  vec_gbb(vsll1);
// CHECK: call <16 x i8> @llvm.ppc.altivec.vgbbd
  dummy();
// CHECK: call void @dummy()
  vec_reve(vsll1);
// CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
  dummy();
// CHECK: call void @dummy()
  vec_revb(vsll1);
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm
}
