// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-LE
#include <altivec.h>

vector bool char vbc = { 0, 1, 0, 1, 0, 1, 0, 1,
                         0, 1, 0, 1, 0, 1, 0, 1 };
vector signed char vsc = { -8,  9, -10, 11, -12, 13, -14, 15,
                           -0,  1,  -2,  3,  -4,  5,  -6,  7};
vector unsigned char vuc = { 8,  9, 10, 11, 12, 13, 14, 15,
                             0,  1,  2,  3,  4,  5,  6,  7};
vector float vf = { -1.5, 2.5, -3.5, 4.5 };
vector double vd = { 3.5, -7.5 };
vector bool short vbs = { 0, 1, 0, 1, 0, 1, 0, 1 };
vector signed short vss = { -1, 2, -3, 4, -5, 6, -7, 8 };
vector unsigned short vus = { 0, 1, 2, 3, 4, 5, 6, 7 };
vector bool int vbi = { 0, 1, 0, 1 };
vector signed int vsi = { -1, 2, -3, 4 };
vector unsigned int vui = { 0, 1, 2, 3 };
vector bool long long vbll = { 1, 0 };
vector signed long long vsll = { 255LL, -937LL };
vector unsigned long long vull = { 1447LL, 2894LL };
double d = 23.4;
signed long long sll = 618LL;
unsigned long long ull = 618ULL;
float af[4] = {23.4f, 56.7f, 89.0f, 12.3f};
double ad[2] = {23.4, 56.7};
signed char asc[16] = { -8,  9, -10, 11, -12, 13, -14, 15,
                        -0,  1,  -2,  3,  -4,  5,  -6,  7};
unsigned char auc[16] = { 8,  9, 10, 11, 12, 13, 14, 15,
                          0,  1,  2,  3,  4,  5,  6,  7};
signed short ass[8] = { -1, 2, -3, 4, -5, 6, -7, 8 };
unsigned short aus[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
signed int asi[4] = { -1, 2, -3, 4 };
unsigned int aui[4] = { 0, 1, 2, 3 };
signed long long asll[2] = { -1L, 2L };
unsigned long long aull[2] = { 1L, 2L };

vector float res_vf;
vector double res_vd;
vector bool char res_vbc;
vector signed char res_vsc;
vector unsigned char res_vuc;
vector bool short res_vbs;
vector signed short res_vss;
vector unsigned short res_vus;
vector bool int res_vbi;
vector signed int res_vsi;
vector unsigned int res_vui;
vector bool long long res_vbll;
vector signed long long res_vsll;
vector unsigned long long res_vull;
vector signed __int128 res_vslll;

double res_d;
int res_i;
float res_af[4];
double res_ad[2];
signed char res_asc[16];
unsigned char res_auc[16];
signed short res_ass[8];
unsigned short res_aus[8];
signed int res_asi[4];
unsigned int res_aui[4];

void dummy() { }

void test1() {
// CHECK-LABEL: define{{.*}} void @test1
// CHECK-LE-LABEL: define{{.*}} void @test1

  res_vf = vec_abs(vf);
// CHECK: call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{[0-9]*}})
// CHECK-LE: call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{[0-9]*}})

  res_vd = vec_abs(vd);
// CHECK: call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{[0-9]*}})
// CHECK-LE: call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{[0-9]*}})

  res_vf = vec_nabs(vf);
// CHECK: [[VEC:%[0-9]+]] = call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{[0-9]*}})
// CHECK-NEXT: fneg <4 x float> [[VEC]]

  res_vd = vec_nabs(vd);
// CHECK: [[VECD:%[0-9]+]] = call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{[0-9]*}})
// CHECK: fneg <2 x double> [[VECD]]

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_add(vd, vd);
// CHECK: fadd <2 x double>
// CHECK-LE: fadd <2 x double>

  res_vd = vec_and(vbll, vd);
// CHECK: and <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>
// CHECK-LE: and <2 x i64>
// CHECK-LE: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>

  res_vd = vec_and(vd, vbll);
// CHECK: and <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>
// CHECK-LE: and <2 x i64>
// CHECK-LE: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>

  res_vd = vec_and(vd, vd);
// CHECK: and <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>
// CHECK-LE: and <2 x i64>
// CHECK-LE: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_andc(vbll, vd);
// CHECK: bitcast <2 x double> %{{[0-9]*}} to <2 x i64>
// CHECK: xor <2 x i64> %{{[0-9]*}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>
// CHECK-LE: bitcast <2 x double> %{{[0-9]*}} to <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]*}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64>
// CHECK-LE: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_andc(vd, vbll);
// CHECK: bitcast <2 x double> %{{[0-9]*}} to <2 x i64>
// CHECK: xor <2 x i64> %{{[0-9]*}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>
// CHECK-LE: bitcast <2 x double> %{{[0-9]*}} to <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]*}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64>
// CHECK-LE: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>

  dummy();
// CHECK: call void @dummy()

  res_vd = vec_andc(vd, vd);
// CHECK: bitcast <2 x double> %{{[0-9]*}} to <2 x i64>
// CHECK: xor <2 x i64> %{{[0-9]*}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]*}} to <2 x double>

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_ceil(vd);
// CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{[0-9]*}})
// CHECK-LE: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{[0-9]*}})

  res_vf = vec_ceil(vf);
// CHECK: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{[0-9]*}})
// CHECK-LE: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{[0-9]*}})

  res_vbll = vec_cmpeq(vd, vd);
// CHECK: call <2 x i64> @llvm.ppc.vsx.xvcmpeqdp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})
// CHECK-LE: call <2 x i64> @llvm.ppc.vsx.xvcmpeqdp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})

  res_vbi = vec_cmpeq(vf, vf);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xvcmpeqsp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xvcmpeqsp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})

  res_vbll = vec_cmpge(vd, vd);
// CHECK: call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})
// CHECK-LE: call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})

  res_vbi = vec_cmpge(vf, vf);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})

  res_vbll = vec_cmpgt(vd, vd);
// CHECK: call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})
// CHECK-LE: call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})

  res_vbi = vec_cmpgt(vf, vf);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})

  res_vbll = vec_cmple(vd, vd);
// CHECK: call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})
// CHECK-LE: call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})

  res_vbi = vec_cmple(vf, vf);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})

  res_vbll = vec_cmplt(vd, vd);
// CHECK: call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})
// CHECK-LE: call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %{{[0-9]*}}, <2 x double> %{{[0-9]*}})

  res_vbi = vec_cmplt(vf, vf);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %{{[0-9]*}}, <4 x float> %{{[0-9]*}})

  /* vec_cpsgn */
  res_vf = vec_cpsgn(vf, vf);
// CHECK: call <4 x float> @llvm.copysign.v4f32(<4 x float> %{{.+}}, <4 x float> %{{.+}})
// CHECK-LE: call <4 x float> @llvm.copysign.v4f32(<4 x float> %{{.+}}, <4 x float> %{{.+}})

  res_vd = vec_cpsgn(vd, vd);
// CHECK: call <2 x double> @llvm.copysign.v2f64(<2 x double> %{{.+}}, <2 x double> %{{.+}})
// CHECK-LE: call <2 x double> @llvm.copysign.v2f64(<2 x double> %{{.+}}, <2 x double> %{{.+}})

  /* vec_div */
  res_vsll = vec_div(vsll, vsll);
// CHECK: sdiv <2 x i64>
// CHECK-LE: sdiv <2 x i64>

  res_vull = vec_div(vull, vull);
// CHECK: udiv <2 x i64>
// CHECK-LE: udiv <2 x i64>

  res_vf = vec_div(vf, vf);
// CHECK: fdiv <4 x float>
// CHECK-LE: fdiv <4 x float>

  res_vd = vec_div(vd, vd);
// CHECK: fdiv <2 x double>
// CHECK-LE: fdiv <2 x double>

  /* vec_max */
  res_vf = vec_max(vf, vf);
// CHECK: @llvm.ppc.vsx.xvmaxsp
// CHECK-LE: @llvm.ppc.vsx.xvmaxsp

  res_vd = vec_max(vd, vd);
// CHECK: @llvm.ppc.vsx.xvmaxdp
// CHECK-LE: @llvm.ppc.vsx.xvmaxdp

  res_vf = vec_vmaxfp(vf, vf);
// CHECK: @llvm.ppc.vsx.xvmaxsp
// CHECK-LE: @llvm.ppc.vsx.xvmaxsp

  /* vec_min */
  res_vf = vec_min(vf, vf);
// CHECK: @llvm.ppc.vsx.xvminsp
// CHECK-LE: @llvm.ppc.vsx.xvminsp

  res_vd = vec_min(vd, vd);
// CHECK: @llvm.ppc.vsx.xvmindp
// CHECK-LE: @llvm.ppc.vsx.xvmindp

  res_vf = vec_vminfp(vf, vf);
// CHECK: @llvm.ppc.vsx.xvminsp
// CHECK-LE: @llvm.ppc.vsx.xvminsp

  res_d = __builtin_vsx_xsmaxdp(d, d);
// CHECK: @llvm.ppc.vsx.xsmaxdp
// CHECK-LE: @llvm.ppc.vsx.xsmaxdp

  res_d = __builtin_vsx_xsmindp(d, d);
// CHECK: @llvm.ppc.vsx.xsmindp
// CHECK-LE: @llvm.ppc.vsx.xsmindp

  /* vec_perm */
  res_vsll = vec_perm(vsll, vsll, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_perm(vull, vull, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbll = vec_perm(vbll, vbll, vuc);
// CHECK: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>

  res_vf = vec_round(vf);
// CHECK: call <4 x float> @llvm.round.v4f32(<4 x float>
// CHECK-LE: call <4 x float> @llvm.round.v4f32(<4 x float>

  res_vd = vec_round(vd);
// CHECK: call <2 x double> @llvm.round.v2f64(<2 x double>
// CHECK-LE: call <2 x double> @llvm.round.v2f64(<2 x double>

  res_vd = vec_perm(vd, vd, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vd = vec_splat(vd, 1);
// CHECK: [[T1:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>

  res_vbll = vec_splat(vbll, 1);
// CHECK: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>

  res_vsll =  vec_splat(vsll, 1);
// CHECK: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>

  res_vull =  vec_splat(vull, 1);
// CHECK: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>

  res_vsi = vec_pack(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_pack(vull, vull);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_pack(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_vperm(vsll, vsll, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_vperm(vull, vull, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vd = vec_vperm(vd, vd, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_vsx_ld */

  res_vbi = vec_vsx_ld(0, &vbi);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vsi = vec_vsx_ld(0, &vsi);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vsi = vec_vsx_ld(0, asi);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vui = vec_vsx_ld(0, &vui);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vui = vec_vsx_ld(0, aui);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vf = vec_vsx_ld (0, &vf);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vf = vec_vsx_ld (0, af);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vsll = vec_vsx_ld(0, &vsll);
// CHECK: @llvm.ppc.vsx.lxvd2x
// CHECK-LE: @llvm.ppc.vsx.lxvd2x

  res_vull = vec_vsx_ld(0, &vull);
// CHECK: @llvm.ppc.vsx.lxvd2x
// CHECK-LE: @llvm.ppc.vsx.lxvd2x

  res_vd = vec_vsx_ld(0, &vd);
// CHECK: @llvm.ppc.vsx.lxvd2x
// CHECK-LE: @llvm.ppc.vsx.lxvd2x

  res_vd = vec_vsx_ld(0, ad);
// CHECK: @llvm.ppc.vsx.lxvd2x
// CHECK-LE: @llvm.ppc.vsx.lxvd2x

  res_vbs = vec_vsx_ld(0, &vbs);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vss = vec_vsx_ld(0, &vss);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vss = vec_vsx_ld(0, ass);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vus = vec_vsx_ld(0, &vus);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vus = vec_vsx_ld(0, aus);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vbc = vec_vsx_ld(0, &vbc);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vsc = vec_vsx_ld(0, &vsc);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vuc = vec_vsx_ld(0, &vuc);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vsc = vec_vsx_ld(0, asc);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vuc = vec_vsx_ld(0, auc);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  /* vec_vsx_st */

  vec_vsx_st(vbi, 0, &res_vbi);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vbi, 0, res_aui);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vbi, 0, res_asi);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vsi, 0, &res_vsi);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vsi, 0, res_asi);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vui, 0, &res_vui);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vui, 0, res_aui);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vf, 0, &res_vf);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vf, 0, res_af);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vsll, 0, &res_vsll);
// CHECK: @llvm.ppc.vsx.stxvd2x
// CHECK-LE: @llvm.ppc.vsx.stxvd2x

  vec_vsx_st(vull, 0, &res_vull);
// CHECK: @llvm.ppc.vsx.stxvd2x
// CHECK-LE: @llvm.ppc.vsx.stxvd2x

  vec_vsx_st(vd, 0, &res_vd);
// CHECK: @llvm.ppc.vsx.stxvd2x
// CHECK-LE: @llvm.ppc.vsx.stxvd2x

  vec_vsx_st(vd, 0, res_ad);
// CHECK: @llvm.ppc.vsx.stxvd2x
// CHECK-LE: @llvm.ppc.vsx.stxvd2x

  vec_vsx_st(vbs, 0, &res_vbs);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vbs, 0, res_aus);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vbs, 0, res_ass);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vss, 0, &res_vss);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vss, 0, res_ass);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vus, 0, &res_vus);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vus, 0, res_aus);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vsc, 0, &res_vsc);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vsc, 0, res_asc);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vuc, 0, &res_vuc);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vuc, 0, res_auc);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vbc, 0, &res_vbc);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vbc, 0, res_asc);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vbc, 0, res_auc);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  /* vec_and */
  res_vsll = vec_and(vsll, vsll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vsll = vec_and(vbll, vsll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vsll = vec_and(vsll, vbll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_and(vull, vull);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_and(vbll, vull);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_and(vull, vbll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vbll = vec_and(vbll, vbll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  /* vec_vand */
  res_vsll = vec_vand(vsll, vsll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vsll = vec_vand(vbll, vsll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vsll = vec_vand(vsll, vbll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_vand(vull, vull);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_vand(vbll, vull);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_vand(vull, vbll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vbll = vec_vand(vbll, vbll);
// CHECK: and <2 x i64>
// CHECK-LE: and <2 x i64>

  /* vec_andc */
  res_vsll = vec_andc(vsll, vsll);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK-LE: xor <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vsll = vec_andc(vbll, vsll);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK-LE: xor <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vsll = vec_andc(vsll, vbll);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK-LE: xor <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_andc(vull, vull);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK-LE: xor <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_andc(vbll, vull);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK-LE: xor <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vull = vec_andc(vull, vbll);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK-LE: xor <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vbll = vec_andc(vbll, vbll);
// CHECK: xor <2 x i64>
// CHECK: and <2 x i64>
// CHECK-LE: xor <2 x i64>
// CHECK-LE: and <2 x i64>

  res_vf = vec_floor(vf);
// CHECK: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{[0-9]+}})

  res_vd = vec_floor(vd);
// CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{[0-9]+}})

  res_vf = vec_madd(vf, vf, vf);
// CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}})

  res_vd = vec_madd(vd, vd, vd);
// CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})

  /* vec_mergeh */
  res_vsll = vec_mergeh(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_mergeh(vsll, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_mergeh(vbll, vsll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_mergeh(vull, vull);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_mergeh(vull, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_mergeh(vbll, vull);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_mergel */
  res_vsll = vec_mergel(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_mergel(vsll, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_mergel(vbll, vsll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_mergel(vull, vull);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_mergel(vull, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_mergel(vbll, vull);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_msub */
  res_vf = vec_msub(vf, vf, vf);
// CHECK: fneg <4 x float> %{{[0-9]+}}
// CHECK-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>
// CHECK-LE: fneg <4 x float> %{{[0-9]+}}
// CHECK-LE-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>

  res_vd = vec_msub(vd, vd, vd);
// CHECK: fneg <2 x double> %{{[0-9]+}}
// CHECK-NEXT: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double>
// CHECK-LE: fneg <2 x double> %{{[0-9]+}}
// CHECK-LE-NEXT: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double>

  res_vsll = vec_mul(vsll, vsll);
// CHECK: mul <2 x i64>
// CHECK-LE: mul <2 x i64>

  res_vull = vec_mul(vull, vull);
// CHECK: mul <2 x i64>
// CHECK-LE: mul <2 x i64>

  res_vf = vec_mul(vf, vf);
// CHECK: fmul <4 x float> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: fmul <4 x float> %{{[0-9]+}}, %{{[0-9]+}}

  res_vd = vec_mul(vd, vd);
// CHECK: fmul <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: fmul <2 x double> %{{[0-9]+}}, %{{[0-9]+}}

  res_vf = vec_nearbyint(vf);
// CHECK: call <4 x float> @llvm.round.v4f32(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.round.v4f32(<4 x float> %{{[0-9]+}})

  res_vd = vec_nearbyint(vd);
// CHECK: call <2 x double> @llvm.round.v2f64(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.round.v2f64(<2 x double> %{{[0-9]+}})

  res_vf = vec_nmadd(vf, vf, vf);
// CHECK: [[FM:[0-9]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}})
// CHECK-NEXT: fneg <4 x float> %[[FM]]
// CHECK-LE: [[FM:[0-9]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}})
// CHECK-LE-NEXT: fneg <4 x float> %[[FM]]

  res_vd = vec_nmadd(vd, vd, vd);
// CHECK: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})
// CHECK-NEXT: fneg <2 x double> %[[FM]]
// CHECK-LE: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})
// CHECK-LE-NEXT: fneg <2 x double> %[[FM]]

  res_vf = vec_nmsub(vf, vf, vf);
// CHECK: fneg <4 x float> %{{[0-9]+}}
// CHECK-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>
// CHECK: fneg <4 x float> %{{[0-9]+}}
// CHECK-LE: fneg <4 x float> %{{[0-9]+}}
// CHECK-LE-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>
// CHECK-LE: fneg <4 x float> %{{[0-9]+}}

  res_vd = vec_nmsub(vd, vd, vd);
// CHECK: fneg <2 x double> %{{[0-9]+}}
// CHECK-NEXT: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double>
// CHECK-NEXT: fneg <2 x double> %[[FM]]
// CHECK-LE: fneg <2 x double> %{{[0-9]+}}
// CHECK-LE-NEXT: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double>
// CHECK-LE-NEXT: fneg <2 x double> %[[FM]]

  /* vec_nor */
  res_vsll = vec_nor(vsll, vsll);
// CHECK: or <2 x i64>
// CHECK: xor <2 x i64>
// CHECK-LE: or <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_nor(vull, vull);
// CHECK: or <2 x i64>
// CHECK: xor <2 x i64>
// CHECK-LE: or <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_nor(vbll, vbll);
// CHECK: or <2 x i64>
// CHECK: xor <2 x i64>
// CHECK-LE: or <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vd = vec_nor(vd, vd);
// CHECK: bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK: [[OR:%.+]] = or <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: xor <2 x i64> [[OR]], <i64 -1, i64 -1>
// CHECK-LE: bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK-LE: [[OR:%.+]] = or <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE-NEXT: xor <2 x i64> [[OR]], <i64 -1, i64 -1>

  /* vec_or */
  res_vsll = vec_or(vsll, vsll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vsll = vec_or(vbll, vsll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vsll = vec_or(vsll, vbll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vull = vec_or(vull, vull);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vull = vec_or(vbll, vull);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vull = vec_or(vull, vbll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vbll = vec_or(vbll, vbll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vd = vec_or(vd, vd);
// CHECK: bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK: or <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK-LE: or <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}

  res_vd = vec_or(vbll, vd);
// CHECK: [[T1:%.+]] = bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK: [[T2:%.+]] = or <2 x i64> %{{[0-9]+}}, [[T1]]
// CHECK: bitcast <2 x i64> [[T2]] to <2 x double>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK-LE: [[T2:%.+]] = or <2 x i64> %{{[0-9]+}}, [[T1]]
// CHECK-LE: bitcast <2 x i64> [[T2]] to <2 x double>

  res_vd = vec_or(vd, vbll);
// CHECK: [[T1:%.+]] = bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK: [[T2:%.+]] = or <2 x i64> [[T1]], %{{[0-9]+}}
// CHECK: bitcast <2 x i64> [[T2]] to <2 x double>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x double> %{{[0-9]+}} to <2 x i64>
// CHECK-LE: [[T2:%.+]] = or <2 x i64> [[T1]], %{{[0-9]+}}
// CHECK-LE: bitcast <2 x i64> [[T2]] to <2 x double>

  res_vf = vec_re(vf);
// CHECK: call <4 x float> @llvm.ppc.vsx.xvresp(<4 x float>
// CHECK-LE: call <4 x float> @llvm.ppc.vsx.xvresp(<4 x float>

  res_vd = vec_re(vd);
// CHECK: call <2 x double> @llvm.ppc.vsx.xvredp(<2 x double>
// CHECK-LE: call <2 x double> @llvm.ppc.vsx.xvredp(<2 x double>

  res_vf = vec_rint(vf);
// CHECK: call <4 x float> @llvm.rint.v4f32(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.rint.v4f32(<4 x float> %{{[0-9]+}})

  res_vd = vec_rint(vd);
// CHECK: call <2 x double> @llvm.rint.v2f64(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.rint.v2f64(<2 x double> %{{[0-9]+}})

  res_vf = vec_rsqrte(vf);
// CHECK: call <4 x float> @llvm.ppc.vsx.xvrsqrtesp(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.ppc.vsx.xvrsqrtesp(<4 x float> %{{[0-9]+}})

  res_vd = vec_rsqrte(vd);
// CHECK: call <2 x double> @llvm.ppc.vsx.xvrsqrtedp(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.ppc.vsx.xvrsqrtedp(<2 x double> %{{[0-9]+}})

  res_i = vec_test_swsqrt(vd);
// CHECK: call i32 @llvm.ppc.vsx.xvtsqrtdp(<2 x double> %{{[0-9]+}})
// CHECK-LE: call i32 @llvm.ppc.vsx.xvtsqrtdp(<2 x double> %{{[0-9]+}})

  res_i = vec_test_swsqrts(vf);
// CHECK: call i32 @llvm.ppc.vsx.xvtsqrtsp(<4 x float> %{{[0-9]+}})
// CHECK-LE: call i32 @llvm.ppc.vsx.xvtsqrtsp(<4 x float> %{{[0-9]+}})

  res_i = vec_test_swdiv(vd, vd);
// CHECK: call i32 @llvm.ppc.vsx.xvtdivdp(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})
// CHECK-LE: call i32 @llvm.ppc.vsx.xvtdivdp(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})

  res_i = vec_test_swdivs(vf, vf);
// CHECK: call i32 @llvm.ppc.vsx.xvtdivsp(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}})
// CHECK-LE: call i32 @llvm.ppc.vsx.xvtdivsp(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}})


  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_sel(vd, vd, vbll);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]+}} to <2 x double>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>
// CHECK-LE: bitcast <2 x i64> %{{[0-9]+}} to <2 x double>

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_sel(vd, vd, vull);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK: bitcast <2 x i64> %{{[0-9]+}} to <2 x double>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>
// CHECK-LE: bitcast <2 x i64> %{{[0-9]+}} to <2 x double>

  res_vbll = vec_sel(vbll, vbll, vbll);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>

  res_vbll = vec_sel(vbll, vbll, vull);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>

  res_vsll = vec_sel(vsll, vsll, vbll);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>

  res_vsll = vec_sel(vsll, vsll, vull);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>

  res_vull = vec_sel(vull, vull, vbll);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>

  res_vull = vec_sel(vull, vull, vull);
// CHECK: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK: and <2 x i64> %{{[0-9]+}},
// CHECK: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: or <2 x i64>
// CHECK-LE: xor <2 x i64> %{{[0-9]+}}, <i64 -1, i64 -1>
// CHECK-LE: and <2 x i64> %{{[0-9]+}},
// CHECK-LE: and <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: or <2 x i64>

  res_vf = vec_sqrt(vf);
// CHECK: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{[0-9]+}})

  res_vd = vec_sqrt(vd);
// CHECK: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{[0-9]+}})

  res_vd = vec_sub(vd, vd);
// CHECK: fsub <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: fsub <2 x double> %{{[0-9]+}}, %{{[0-9]+}}

  res_vf = vec_trunc(vf);
// CHECK: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{[0-9]+}})

  res_vd = vec_trunc(vd);
// CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{[0-9]+}})

  /* vec_vor */
  res_vsll = vec_vor(vsll, vsll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vsll = vec_vor(vbll, vsll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vsll = vec_vor(vsll, vbll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vull = vec_vor(vull, vull);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vull = vec_vor(vbll, vull);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vull = vec_vor(vull, vbll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  res_vbll = vec_vor(vbll, vbll);
// CHECK: or <2 x i64>
// CHECK-LE: or <2 x i64>

  /* vec_xor */
  res_vsll = vec_xor(vsll, vsll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vsll = vec_xor(vbll, vsll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vsll = vec_xor(vsll, vbll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_xor(vull, vull);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_xor(vbll, vull);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_xor(vull, vbll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vbll = vec_xor(vbll, vbll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_xor(vd, vd);
// CHECK: [[X1:%.+]] = xor <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: bitcast <2 x i64> [[X1]] to <2 x double>
// CHECK-LE: [[X1:%.+]] = xor <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: bitcast <2 x i64> [[X1]] to <2 x double>

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_xor(vd, vbll);
// CHECK: [[X1:%.+]] = xor <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: bitcast <2 x i64> [[X1]] to <2 x double>
// CHECK-LE: [[X1:%.+]] = xor <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: bitcast <2 x i64> [[X1]] to <2 x double>

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vd = vec_xor(vbll, vd);
// CHECK: [[X1:%.+]] = xor <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK: bitcast <2 x i64> [[X1]] to <2 x double>
// CHECK-LE: [[X1:%.+]] = xor <2 x i64> %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-LE: bitcast <2 x i64> [[X1]] to <2 x double>

  /* vec_vxor */
  res_vsll = vec_vxor(vsll, vsll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vsll = vec_vxor(vbll, vsll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vsll = vec_vxor(vsll, vbll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_vxor(vull, vull);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_vxor(vbll, vull);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vull = vec_vxor(vull, vbll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vbll = vec_vxor(vbll, vbll);
// CHECK: xor <2 x i64>
// CHECK-LE: xor <2 x i64>

  res_vsll = vec_cts(vd, 0);
// CHECK: fmul <2 x double>
// CHECK: fptosi <2 x double> %{{.*}} to <2 x i64>
// CHECK-LE: fmul <2 x double>
// CHECK-LE: fptosi <2 x double> %{{.*}} to <2 x i64>

  res_vsll = vec_cts(vd, 31);
// CHECK: fmul <2 x double>
// CHECK: fptosi <2 x double> %{{.*}} to <2 x i64>
// CHECK-LE: fmul <2 x double>
// CHECK-LE: fptosi <2 x double> %{{.*}} to <2 x i64>

  res_vsll = vec_ctu(vd, 0);
// CHECK: fmul <2 x double>
// CHECK: fptoui <2 x double> %{{.*}} to <2 x i64>
// CHECK-LE: fmul <2 x double>
// CHECK-LE: fptoui <2 x double> %{{.*}} to <2 x i64>

  res_vsll = vec_ctu(vd, 31);
// CHECK: fmul <2 x double>
// CHECK: fptoui <2 x double> %{{.*}} to <2 x i64>
// CHECK-LE: fmul <2 x double>
// CHECK-LE: fptoui <2 x double> %{{.*}} to <2 x i64>

  res_vd = vec_ctf(vsll, 0);
// CHECK: sitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK: fmul <2 x double>
// CHECK-LE: sitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK-LE: fmul <2 x double>

  res_vd = vec_ctf(vsll, 31);
// CHECK: sitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK: fmul <2 x double>
// CHECK-LE: sitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK-LE: fmul <2 x double>

  res_vd = vec_ctf(vull, 0);
// CHECK: uitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK: fmul <2 x double>
// CHECK-LE: uitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK-LE: fmul <2 x double>

  res_vd = vec_ctf(vull, 31);
// CHECK: uitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK: fmul <2 x double>
// CHECK-LE: uitofp <2 x i64> %{{.*}} to <2 x double>
// CHECK-LE: fmul <2 x double>

  res_vsll = vec_signed(vd);
// CHECK: fptosi <2 x double>
// CHECK-LE: fptosi <2 x double>

  res_vsi = vec_signed2(vd, vd);
// CHECK: extractelement <2 x double>
// CHECK: fptosi double
// CHECK: insertelement <4 x i32>
// CHECK: extractelement <2 x double>
// CHECK: fptosi double
// CHECK: insertelement <4 x i32>
// CHECK: extractelement <2 x double>
// CHECK: fptosi double
// CHECK: insertelement <4 x i32>
// CHECK: extractelement <2 x double>
// CHECK: fptosi double
// CHECK: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptosi double
// CHECK-LE: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptosi double
// CHECK-LE: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptosi double
// CHECK-LE: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptosi double
// CHECK-LE: insertelement <4 x i32>

  res_vsi = vec_signede(vd);
// CHECK: @llvm.ppc.vsx.xvcvdpsxws(<2 x double>
// CHECK-LE: @llvm.ppc.vsx.xvcvdpsxws(<2 x double>
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsi = vec_signedo(vd);
// CHECK: @llvm.ppc.vsx.xvcvdpsxws(<2 x double>
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvdpsxws(<2 x double>

  res_vull = vec_unsigned(vd);
// CHECK: fptoui <2 x double>
// CHECK-LE: fptoui <2 x double>

  res_vui = vec_unsigned2(vd, vd);
// CHECK: extractelement <2 x double>
// CHECK: fptoui double
// CHECK: insertelement <4 x i32>
// CHECK: extractelement <2 x double>
// CHECK: fptoui double
// CHECK: insertelement <4 x i32>
// CHECK: extractelement <2 x double>
// CHECK: fptoui double
// CHECK: insertelement <4 x i32>
// CHECK: extractelement <2 x double>
// CHECK: fptoui double
// CHECK: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptoui double
// CHECK-LE: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptoui double
// CHECK-LE: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptoui double
// CHECK-LE: insertelement <4 x i32>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptoui double
// CHECK-LE: insertelement <4 x i32>

  res_vui = vec_unsignede(vd);
// CHECK: @llvm.ppc.vsx.xvcvdpuxws(<2 x double>
// CHECK-LE: @llvm.ppc.vsx.xvcvdpuxws(<2 x double>
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_unsignedo(vd);
// CHECK: @llvm.ppc.vsx.xvcvdpuxws(<2 x double>
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvdpuxws(<2 x double>

  res_vf = vec_float2(vsll, vsll);
// CHECK: extractelement <2 x i64>
// CHECK: sitofp i64
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x i64>
// CHECK: sitofp i64
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x i64>
// CHECK: sitofp i64
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x i64>
// CHECK: sitofp i64
// CHECK: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: sitofp i64
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: sitofp i64
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: sitofp i64
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: sitofp i64
// CHECK-LE: insertelement <4 x float>

  res_vf = vec_float2(vull, vull);
// CHECK: extractelement <2 x i64>
// CHECK: uitofp i64
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x i64>
// CHECK: uitofp i64
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x i64>
// CHECK: uitofp i64
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x i64>
// CHECK: uitofp i64
// CHECK: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: uitofp i64
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: uitofp i64
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: uitofp i64
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x i64>
// CHECK-LE: uitofp i64
// CHECK-LE: insertelement <4 x float>

  res_vf = vec_float2(vd, vd);
// CHECK: extractelement <2 x double>
// CHECK: fptrunc double
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x double>
// CHECK: fptrunc double
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x double>
// CHECK: fptrunc double
// CHECK: insertelement <4 x float>
// CHECK: extractelement <2 x double>
// CHECK: fptrunc double
// CHECK: insertelement <4 x float>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptrunc double
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptrunc double
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptrunc double
// CHECK-LE: insertelement <4 x float>
// CHECK-LE: extractelement <2 x double>
// CHECK-LE: fptrunc double
// CHECK-LE: insertelement <4 x float>

  res_vf = vec_floate(vsll);
// CHECK: @llvm.ppc.vsx.xvcvsxdsp
// CHECK-LE: @llvm.ppc.vsx.xvcvsxdsp
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf = vec_floate(vull);
// CHECK: @llvm.ppc.vsx.xvcvuxdsp
// CHECK-LE: @llvm.ppc.vsx.xvcvuxdsp
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf = vec_floate(vd);
// CHECK: @llvm.ppc.vsx.xvcvdpsp
// CHECK-LE: @llvm.ppc.vsx.xvcvdpsp
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf = vec_floato(vsll);
// CHECK: @llvm.ppc.vsx.xvcvsxdsp
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvsxdsp

  res_vf = vec_floato(vull);
// CHECK: @llvm.ppc.vsx.xvcvuxdsp
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvuxdsp

  res_vf = vec_floato(vd);
// CHECK: @llvm.ppc.vsx.xvcvdpsp
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvdpsp

  res_vd = vec_double(vsll);
// CHECK: sitofp <2 x i64>
// CHECK-LE: sitofp <2 x i64>

  res_vd = vec_double(vull);
// CHECK: uitofp <2 x i64>
// CHECK-LE: uitofp <2 x i64>

  res_vd = vec_doublee(vsi);
// CHECK: @llvm.ppc.vsx.xvcvsxwdp(<4 x i32
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvsxwdp(<4 x i32

  res_vd = vec_doublee(vui);
// CHECK: @llvm.ppc.vsx.xvcvuxwdp(<4 x i32
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvuxwdp(<4 x i32

  res_vd = vec_doublee(vf);
// CHECK: @llvm.ppc.vsx.xvcvspdp(<4 x float
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.vsx.xvcvspdp(<4 x float

  res_vd = vec_doubleh(vsi);
// CHECK: extractelement <4 x i32>
// CHECK: sitofp i32
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x i32>
// CHECK: sitofp i32
// CHECK: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: sitofp i32
// CHECK-LE: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: sitofp i32
// CHECK-LE: insertelement <2 x double>

  res_vd = vec_doubleh(vui);
// CHECK: extractelement <4 x i32>
// CHECK: uitofp i32
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x i32>
// CHECK: uitofp i32
// CHECK: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: uitofp i32
// CHECK-LE: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: uitofp i32
// CHECK-LE: insertelement <2 x double>

  res_vd = vec_doubleh(vf);
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
// CHECK-LE: extractelement <4 x float>
// CHECK-LE: fpext float
// CHECK-LE: insertelement <2 x double>
// CHECK-LE: extractelement <4 x float>
// CHECK-LE: fpext float
// CHECK-LE: insertelement <2 x double>

  res_vd = vec_doublel(vsi);
// CHECK: extractelement <4 x i32>
// CHECK: sitofp i32
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x i32>
// CHECK: sitofp i32
// CHECK: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: sitofp i32
// CHECK-LE: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: sitofp i32
// CHECK-LE: insertelement <2 x double>

  res_vd = vec_doublel(vui);
// CHECK: extractelement <4 x i32>
// CHECK: uitofp i32
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x i32>
// CHECK: uitofp i32
// CHECK: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: uitofp i32
// CHECK-LE: insertelement <2 x double>
// CHECK-LE: extractelement <4 x i32>
// CHECK-LE: uitofp i32
// CHECK-LE: insertelement <2 x double>

  res_vd = vec_doublel(vf);
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
// CHECK-LE: extractelement <4 x float>
// CHECK-LE: fpext float
// CHECK-LE: insertelement <2 x double>
// CHECK-LE: extractelement <4 x float>
// CHECK-LE: fpext float
// CHECK-LE: insertelement <2 x double>

  res_vd = vec_doubleo(vsi);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.vsx.xvcvsxwdp(<4 x i32>
// CHECK-LE: @llvm.ppc.vsx.xvcvsxwdp(<4 x i32>

  res_vd = vec_doubleo(vui);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.vsx.xvcvuxwdp(<4 x i32>
// CHECK-LE: @llvm.ppc.vsx.xvcvuxwdp(<4 x i32>

  res_vd = vec_doubleo(vf);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.vsx.xvcvspdp(<4 x float>
// CHECK-LE: @llvm.ppc.vsx.xvcvspdp(<4 x float>

  res_vbll = vec_reve(vbll);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>

  res_vsll = vec_reve(vsll);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>

  res_vull = vec_reve(vull);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>

  res_vd = vec_reve(vd);
// CHECK: shufflevector <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>
// CHECK-LE: shufflevector <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x i32> <i32 1, i32 0>

  res_vbll = vec_revb(vbll);
// CHECK: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: store <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vsll = vec_revb(vsll);
// CHECK: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: store <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vull = vec_revb(vull);
// CHECK: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: store <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vd = vec_revb(vd);
// CHECK: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: store <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8>* {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vbll = vec_sld(vbll, vbll, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_sld(vsll, vsll, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_sld(vull, vull, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vd = vec_sld(vd, vd, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_sldw(vsll, vsll, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_sldw(vull, vull, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_sll(vsll, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

res_vull = vec_sll(vull, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

res_vsll = vec_slo(vsll, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vsll = vec_slo(vsll, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vull = vec_slo(vull, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vull = vec_slo(vull, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vsll = vec_srl(vsll, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vull = vec_srl(vull, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vsll = vec_sro(vsll, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vsll = vec_sro(vsll, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vull = vec_sro(vull, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vull = vec_sro(vull, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

res_vsll = vec_xl(sll, asll);
// CHECK: load <2 x i64>, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: load <2 x i64>, <2 x i64>* %{{[0-9]+}}, align 1

res_vull = vec_xl(sll, aull);
// CHECK: load <2 x i64>, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: load <2 x i64>, <2 x i64>* %{{[0-9]+}}, align 1

res_vd = vec_xl(sll, ad);
// CHECK: load <2 x double>, <2 x double>* %{{[0-9]+}}, align 1
// CHECK-LE: load <2 x double>, <2 x double>* %{{[0-9]+}}, align 1

vec_xst(vsll, sll, asll);
// CHECK: store <2 x i64> %{{[0-9]+}}, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: store <2 x i64> %{{[0-9]+}}, <2 x i64>* %{{[0-9]+}}, align 1

vec_xst(vull, sll, aull);
// CHECK: store <2 x i64> %{{[0-9]+}}, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: store <2 x i64> %{{[0-9]+}}, <2 x i64>* %{{[0-9]+}}, align 1

vec_xst(vd, sll, ad);
// CHECK: store <2 x double> %{{[0-9]+}}, <2 x double>* %{{[0-9]+}}, align 1
// CHECK-LE: store <2 x double> %{{[0-9]+}}, <2 x double>* %{{[0-9]+}}, align 1

res_vsll = vec_xl_be(sll, asll);
// CHECK: load <2 x i64>, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: call <2 x double> @llvm.ppc.vsx.lxvd2x.be(i8* %{{[0-9]+}})

res_vull = vec_xl_be(sll, aull);
// CHECK: load <2 x i64>, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: call <2 x double> @llvm.ppc.vsx.lxvd2x.be(i8* %{{[0-9]+}})

res_vd = vec_xl_be(sll, ad);
// CHECK: load <2 x double>, <2 x double>* %{{[0-9]+}}, align 1
// CHECK-LE: call <2 x double> @llvm.ppc.vsx.lxvd2x.be(i8* %{{[0-9]+}})

vec_xst_be(vsll, sll, asll);
// CHECK: store <2 x i64> %{{[0-9]+}}, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: call void @llvm.ppc.vsx.stxvd2x.be(<2 x double> %{{[0-9]+}}, i8* %{{[0-9]+}})

vec_xst_be(vull, sll, aull);
// CHECK: store <2 x i64> %{{[0-9]+}}, <2 x i64>* %{{[0-9]+}}, align 1
// CHECK-LE: call void @llvm.ppc.vsx.stxvd2x.be(<2 x double> %{{[0-9]+}}, i8* %{{[0-9]+}})

vec_xst_be(vd, sll, ad);
// CHECK: store <2 x double> %{{[0-9]+}}, <2 x double>* %{{[0-9]+}}, align 1
// CHECK-LE: call void @llvm.ppc.vsx.stxvd2x.be(<2 x double> %{{[0-9]+}}, i8* %{{[0-9]+}})

  res_vf = vec_neg(vf);
// CHECK: fneg <4 x float> {{%[0-9]+}}
// CHECK-LE: fneg <4 x float> {{%[0-9]+}}

  res_vd = vec_neg(vd);
// CHECK: fneg <2 x double> {{%[0-9]+}}
// CHECK-LE: fneg <2 x double> {{%[0-9]+}}

res_vd = vec_xxpermdi(vd, vd, 0);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>

res_vf = vec_xxpermdi(vf, vf, 1);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 3>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 3>

res_vsll = vec_xxpermdi(vsll, vsll, 2);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 2>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 2>

res_vull = vec_xxpermdi(vull, vull, 3);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 3>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 3>

res_vsi = vec_xxpermdi(vsi, vsi, 0);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>

res_vui = vec_xxpermdi(vui, vui, 1);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 3>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 3>

res_vss = vec_xxpermdi(vss, vss, 2);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 2>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 2>

res_vus = vec_xxpermdi(vus, vus, 3);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 3>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 1, i32 3>

res_vsc = vec_xxpermdi(vsc, vsc, 0);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>

res_vuc = vec_xxpermdi(vuc, vuc, 1);
// CHECK: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 3>
// CHECK-LE: shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 3>

res_vd = vec_xxsldwi(vd, vd, 0);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

res_vf = vec_xxsldwi(vf, vf, 1);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 7, i32 0, i32 1, i32 2>

res_vsll = vec_xxsldwi(vsll, vsll, 2);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 6, i32 7, i32 0, i32 1>

res_vull = vec_xxsldwi(vull, vull, 3);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 5, i32 6, i32 7, i32 0>

res_vsi = vec_xxsldwi(vsi, vsi, 0);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

res_vui = vec_xxsldwi(vui, vui, 1);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 7, i32 0, i32 1, i32 2>

res_vss = vec_xxsldwi(vss, vss, 2);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 6, i32 7, i32 0, i32 1>


res_vus = vec_xxsldwi(vus, vus, 3);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 5, i32 6, i32 7, i32 0>

res_vsc = vec_xxsldwi(vsc, vsc, 0);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

res_vuc = vec_xxsldwi(vuc, vuc, 1);
// CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
// CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 7, i32 0, i32 1, i32 2>

res_vd = vec_promote(d, 0);
// CHECK: store <2 x double> zeroinitializer
// CHECK: insertelement <2 x double>
// CHECK-LE: store <2 x double> zeroinitializer
// CHECK-LE: insertelement <2 x double>

res_vsll = vec_promote(sll, 0);
// CHECK: store <2 x i64> zeroinitializer
// CHECK: insertelement <2 x i64>
// CHECK-LE: store <2 x i64> zeroinitializer
// CHECK-LE: insertelement <2 x i64>

res_vull = vec_promote(ull, 0);
// CHECK: store <2 x i64> zeroinitializer
// CHECK: insertelement <2 x i64>
// CHECK-LE: store <2 x i64> zeroinitializer
// CHECK-LE: insertelement <2 x i64>
}

// The return type of the call expression may be different from the return type of the shufflevector.
// Wrong implementation could crash the compiler, add this test case to check that and avoid ICE.
vector int xxpermdi_should_not_assert(vector int a, vector int b) {
  return vec_xxpermdi(a, b, 0);
// CHECK-LABEL: xxpermdi_should_not_assert
// CHECK:  bitcast <4 x i32> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT:  bitcast <4 x i32> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT:  shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>
// CHECK-NEXT:  bitcast <2 x i64> %{{[0-9]+}} to <4 x i32>

// CHECK-LE:  bitcast <4 x i32> %{{[0-9]+}} to <2 x i64>
// CHECK-LE-NEXT:  bitcast <4 x i32> %{{[0-9]+}} to <2 x i64>
// CHECK-LE-NEXT:  shufflevector <2 x i64> %{{[0-9]+}}, <2 x i64> %{{[0-9]+}}, <2 x i32> <i32 0, i32 2>
// CHECK-LE-NEXT:  bitcast <2 x i64> %{{[0-9]+}} to <4 x i32>
}

vector double xxsldwi_should_not_assert(vector double a, vector double b) {
  return vec_xxsldwi(a, b, 0);
// CHECK-LABEL: xxsldwi_should_not_assert
// CHECK:  bitcast <2 x double> %{{[0-9]+}} to <4 x i32>
// CHECK-NEXT:  bitcast <2 x double> %{{[0-9]+}} to <4 x i32>
// CHECK-NEXT:  shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:  bitcast <4 x i32> %{{[0-9]+}} to <2 x double>

// CHECK-LE:  bitcast <2 x double> %{{[0-9]+}} to <4 x i32>
// CHECK-NEXT-LE:  bitcast <2 x double> %{{[0-9]+}} to <4 x i32>
// CHECK-NEXT-LE:  shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT-LE:  bitcast <4 x i32> %{{[0-9]+}} to <2 x double>
}

void testVectorInt128Pack(){
// CHECK-LABEL: testVectorInt128Pack
// CHECK-LABEL-LE: testVectorInt128Pack
  res_vslll = __builtin_pack_vector_int128(aull[0], aull[1]);
// CHECK: %[[V1:[0-9]+]] = insertelement <2 x i64> undef, i64 %{{[0-9]+}}, i64 0
// CHECK-NEXT: %[[V2:[0-9]+]] = insertelement <2 x i64> %[[V1]], i64 %{{[0-9]+}}, i64 1
// CHECK-NEXT:  bitcast <2 x i64> %[[V2]] to <1 x i128>

// CHECK-LE: %[[V1:[0-9]+]] = insertelement <2 x i64> undef, i64 %{{[0-9]+}}, i64 1
// CHECK-NEXT-LE: %[[V2:[0-9]+]] = insertelement <2 x i64> %[[V1]], i64 %{{[0-9]+}}, i64 0
// CHECK-NEXT-LE:  bitcast <2 x i64> %[[V2]] to <1 x i128>

  __builtin_unpack_vector_int128(res_vslll, 0);
// CHECK:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 0

// CHECK-LE:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT-LE: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 1

  __builtin_unpack_vector_int128(res_vslll, 1);
// CHECK:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 1

// CHECK-LE:  %[[V1:[0-9]+]] = bitcast <1 x i128> %{{[0-9]+}} to <2 x i64>
// CHECK-NEXT-LE: %{{[0-9]+}} = extractelement <2 x i64> %[[V1]], i32 0

}

void test_vector_cpsgn_float(vector float a, vector float b) {
// CHECK-LABEL: test_vector_cpsgn_float
// CHECK-DAG: load{{.*}}%__a
// CHECK-DAG: load{{.*}}%__b
// CHECK-NOT: SEPARATOR
// CHECK-DAG: [[RA:%[0-9]+]] = load <4 x float>, <4 x float>* %__a.addr
// CHECK-DAG: [[RB:%[0-9]+]] = load <4 x float>, <4 x float>* %__b.addr
// CHECK-NEXT: call <4 x float> @llvm.copysign.v4f32(<4 x float> [[RB]], <4 x float> [[RA]])
  vec_cpsgn(a, b);
}

void test_vector_cpsgn_double(vector double a, vector double b) {
// CHECK-LABEL: test_vector_cpsgn_double
// CHECK-DAG: load{{.*}}%__a
// CHECK-DAG: load{{.*}}%__b
// CHECK-NOT: SEPARATOR
// CHECK-DAG: [[RA:%[0-9]+]] = load <2 x double>, <2 x double>* %__a.addr
// CHECK-DAG: [[RB:%[0-9]+]] = load <2 x double>, <2 x double>* %__b.addr
// CHECK-NEXT: call <2 x double> @llvm.copysign.v2f64(<2 x double> [[RB]], <2 x double> [[RA]])
  vec_cpsgn(a, b);
}

void test_builtin_xvcpsgnsp(vector float a, vector float b) {
// CHECK-LABEL: test_builtin_xvcpsgnsp
// CHECK-DAG: load{{.*}}%a
// CHECK-DAG: load{{.*}}%b
// CHECK-NOT: SEPARATOR
// CHECK-DAG: [[RA:%[0-9]+]] = load <4 x float>, <4 x float>* %a.addr
// CHECK-DAG: [[RB:%[0-9]+]] = load <4 x float>, <4 x float>* %b.addr
// CHECK-NEXT: call <4 x float> @llvm.copysign.v4f32(<4 x float> [[RA]], <4 x float> [[RB]])
  __builtin_vsx_xvcpsgnsp(a, b);
}

void test_builtin_xvcpsgndp(vector double a, vector double b) {
// CHECK-LABEL: test_builtin_xvcpsgndp
// CHECK-DAG: load{{.*}}%a
// CHECK-DAG: load{{.*}}%b
// CHECK-NOT: SEPARATOR
// CHECK-DAG: [[RA:%[0-9]+]] = load <2 x double>, <2 x double>* %a.addr
// CHECK-DAG: [[RB:%[0-9]+]] = load <2 x double>, <2 x double>* %b.addr
// CHECK-NEXT: call <2 x double> @llvm.copysign.v2f64(<2 x double> [[RA]], <2 x double> [[RB]])
  __builtin_vsx_xvcpsgndp(a, b);
}
