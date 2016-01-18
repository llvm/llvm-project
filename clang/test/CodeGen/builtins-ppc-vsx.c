// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -faltivec -target-feature +vsx -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -faltivec -target-feature +vsx -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-LE

vector unsigned char vuc = { 8,  9, 10, 11, 12, 13, 14, 15,
                             0,  1,  2,  3,  4,  5,  6,  7};
vector float vf = { -1.5, 2.5, -3.5, 4.5 };
vector double vd = { 3.5, -7.5 };
vector signed int vsi = { -1, 2, -3, 4 };
vector unsigned int vui = { 0, 1, 2, 3 };
vector bool long long vbll = { 1, 0 };
vector signed long long vsll = { 255LL, -937LL };
vector unsigned long long vull = { 1447LL, 2894LL };
double d = 23.4;

vector float res_vf;
vector double res_vd;
vector signed int res_vsi;
vector unsigned int res_vui;
vector bool int res_vbi;
vector bool long long res_vbll;
vector signed long long res_vsll;
vector unsigned long long res_vull;
double res_d;

void dummy() { }

void test1() {
// CHECK-LABEL: define void @test1
// CHECK-LE-LABEL: define void @test1

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

  res_vsi = vec_vsx_ld(0, &vsi);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vui = vec_vsx_ld(0, &vui);
// CHECK: @llvm.ppc.vsx.lxvw4x
// CHECK-LE: @llvm.ppc.vsx.lxvw4x

  res_vf = vec_vsx_ld (0, &vf);
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

  /* vec_vsx_st */

  vec_vsx_st(vsi, 0, &res_vsi);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vui, 0, &res_vui);
// CHECK: @llvm.ppc.vsx.stxvw4x
// CHECK-LE: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vf, 0, &res_vf);
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
// CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{[0-9]+}}
// CHECK-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>
// CHECK-LE: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{[0-9]+}}
// CHECK-LE-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>

  res_vd = vec_msub(vd, vd, vd);
// CHECK: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{[0-9]+}}
// CHECK-NEXT: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double>
// CHECK-LE: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{[0-9]+}}
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
// CHECK-NEXT: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %[[FM]]
// CHECK-LE: [[FM:[0-9]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}})
// CHECK-LE-NEXT: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %[[FM]]

  res_vd = vec_nmadd(vd, vd, vd);
// CHECK: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})
// CHECK-NEXT: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %[[FM]]
// CHECK-LE: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}})
// CHECK-LE-NEXT: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %[[FM]]

  res_vf = vec_nmsub(vf, vf, vf);
// CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{[0-9]+}}
// CHECK-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>
// CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{[0-9]+}}
// CHECK-LE: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{[0-9]+}}
// CHECK-LE-NEXT: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x float>
// CHECK-LE: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{[0-9]+}}

  res_vd = vec_nmsub(vd, vd, vd);
// CHECK: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{[0-9]+}}
// CHECK-NEXT: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double>
// CHECK-NEXT: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %[[FM]]
// CHECK-LE: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{[0-9]+}}
// CHECK-LE-NEXT: [[FM:[0-9]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{[0-9]+}}, <2 x double> %{{[0-9]+}}, <2 x double>
// CHECK-LE-NEXT: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %[[FM]]

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
// CHECK: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %{{[0-9]+}})

  res_vd = vec_rint(vd);
// CHECK: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %{{[0-9]+}})

  res_vf = vec_rsqrte(vf);
// CHECK: call <4 x float> @llvm.ppc.vsx.xvrsqrtesp(<4 x float> %{{[0-9]+}})
// CHECK-LE: call <4 x float> @llvm.ppc.vsx.xvrsqrtesp(<4 x float> %{{[0-9]+}})

  res_vd = vec_rsqrte(vd);
// CHECK: call <2 x double> @llvm.ppc.vsx.xvrsqrtedp(<2 x double> %{{[0-9]+}})
// CHECK-LE: call <2 x double> @llvm.ppc.vsx.xvrsqrtedp(<2 x double> %{{[0-9]+}})

  dummy();
// CHECK: call void @dummy()
// CHECK-LE: call void @dummy()

  res_vf = vec_sel(vd, vd, vbll);
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

}
