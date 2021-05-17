// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +power8-vector -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +power8-vector -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-LE
// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx -triple powerpc64-unknown-unknown -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PPC
// Added -target-feature +vsx above to avoid errors about "vector double" and to
// generate the correct errors for functions that are only overloaded with VSX
// (vec_cmpge, vec_cmple). Without this option, there is only one overload so
// it is selected.
#include <altivec.h>

void dummy() { }
signed int si;
signed long long sll;
unsigned long long ull;
signed __int128 sx;
unsigned __int128 ux;
double d;
vector signed char vsc = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5 };
vector unsigned char vuc = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5 };
vector bool char vbc = { 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1 };

vector signed short vss = { 0, 1, 2, 3, 4, 5, 6, 7 };
vector unsigned short vus = { 0, 1, 2, 3, 4, 5, 6, 7 };
vector bool short vbs = { 1, 1, 0, 0, 0, 0, 1, 1 };

vector signed int vsi = { -1, 2, -3, 4 };
vector unsigned int vui = { 1, 2, 3, 4 };
vector bool int vbi = {0, -1, -1, 0};

vector signed long long vsll = { 1, 2 };
vector unsigned long long vull = { 1, 2 };
vector bool long long vbll = { 1, 0 };

vector signed __int128 vsx = { 1 };
vector unsigned __int128 vux = { 1 };

vector float vfa = { 1.e-4f, -132.23f, -22.1, 32.00f };
vector double vda = { 1.e-11, -132.23e10 };

int res_i;
double res_d;
signed long long res_sll;
unsigned long long res_ull;

vector signed char res_vsc;
vector unsigned char res_vuc;
vector bool char res_vbc;

vector signed short res_vss;
vector unsigned short res_vus;
vector bool short res_vbs;

vector signed int res_vsi;
vector unsigned int res_vui;
vector bool int res_vbi;

vector signed long long res_vsll;
vector unsigned long long res_vull;
vector bool long long res_vbll;

vector signed __int128 res_vsx;
vector unsigned __int128 res_vux;

vector float res_vf;
vector double res_vd;

// CHECK-LABEL: define{{.*}} void @test1
void test1() {

  /* vec_abs */
  res_vsll = vec_abs(vsll);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vmaxsd(<2 x i64> %{{[0-9]*}}, <2 x i64>
// CHECK-LE: call <2 x i64> @llvm.ppc.altivec.vmaxsd(<2 x i64> %{{[0-9]*}}, <2 x i64>
// CHECK-PPC: error: call to 'vec_abs' is ambiguous

  /* vec_add */
  res_vsll = vec_add(vsll, vsll);
// CHECK: add <2 x i64>
// CHECK-LE: add <2 x i64>

  res_vull = vec_add(vull, vull);
// CHECK: add <2 x i64>
// CHECK-LE: add <2 x i64>

  res_vuc = vec_add_u128(vuc, vuc);
// CHECK: add <1 x i128>
// CHECK-LE: add <1 x i128>

  /* vec_addc */
  res_vsi = vec_addc(vsi, vsi);
// CHECK: @llvm.ppc.altivec.vaddcuw
// CHECK-LE: @llvm.ppc.altivec.vaddcuw

  res_vui = vec_addc(vui, vui);
// CHECK: @llvm.ppc.altivec.vaddcuw
// CHECK-LE: @llvm.ppc.altivec.vaddcuw

  res_vsx = vec_addc(vsx, vsx);
// CHECK: @llvm.ppc.altivec.vaddcuq
// CHECK-LE: @llvm.ppc.altivec.vaddcuq

  res_vux = vec_addc(vux, vux);
// CHECK: @llvm.ppc.altivec.vaddcuq
// CHECK-LE: @llvm.ppc.altivec.vaddcuq

  res_vuc = vec_addc_u128(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vaddcuq
// CHECK-LE: @llvm.ppc.altivec.vaddcuq

  /* vec_adde */
  res_vsx = vec_adde(vsx, vsx, vsx);
// CHECK: @llvm.ppc.altivec.vaddeuqm
// CHECK-LE: @llvm.ppc.altivec.vaddeuqm

  res_vux = vec_adde(vux, vux, vux);
// CHECK: @llvm.ppc.altivec.vaddeuqm
// CHECK-LE: @llvm.ppc.altivec.vaddeuqm

  res_vuc = vec_adde_u128(vuc, vuc, vuc);
// CHECK: @llvm.ppc.altivec.vaddeuqm
// CHECK-LE: @llvm.ppc.altivec.vaddeuqm

  /* vec_addec */
  res_vsx = vec_addec(vsx, vsx, vsx);
// CHECK: @llvm.ppc.altivec.vaddecuq
// CHECK-LE: @llvm.ppc.altivec.vaddecuq

  res_vuc = vec_addec_u128(vuc, vuc, vuc);
// CHECK: @llvm.ppc.altivec.vaddecuq
// CHECK-LE: @llvm.ppc.altivec.vaddecuq

  /* vec_mergee */  
  res_vbi = vec_mergee(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm
  
  res_vsi = vec_mergee(vsi, vsi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_mergee(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-PPC: warning: implicit declaration of function 'vec_mergee'

  res_vbll = vec_mergee(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsll = vec_mergee(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vull = vec_mergee(vull, vull);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf = vec_mergee(vfa, vfa);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vd = vec_mergee(vda, vda);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_mergeo */
  res_vbi = vec_mergeo(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsi = vec_mergeo(vsi, vsi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_mergeo(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-PPC: warning: implicit declaration of function 'vec_mergeo'
  
  /* vec_cmpeq */
  res_vbll = vec_cmpeq(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd
// CHECK-LE: @llvm.ppc.altivec.vcmpequd

  res_vbll = vec_cmpeq(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vcmpequd
// CHECK-LE: @llvm.ppc.altivec.vcmpequd

  res_vbll = vec_cmpeq(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd
// CHECK-LE: @llvm.ppc.altivec.vcmpequd

  /* vec_cmpge */
  res_vbll = vec_cmpge(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd

  res_vbll = vec_cmpge(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud

  /* vec_cmple */
  res_vbll = vec_cmple(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd

  res_vbll = vec_cmple(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud

  /* vec_cmpgt */
  res_vbll = vec_cmpgt(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd

  res_vbll = vec_cmpgt(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud

  /* vec_cmplt */
  res_vbll = vec_cmplt(vsll, vsll);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %{{[0-9]*}}, <2 x i64> %{{[0-9]*}})
// CHECK-LE: call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %{{[0-9]*}}, <2 x i64> %{{[0-9]*}})

  res_vbll = vec_cmplt(vull, vull);
// CHECK: call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %{{[0-9]*}}, <2 x i64> %{{[0-9]*}})
// CHECK-LE: call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %{{[0-9]*}}, <2 x i64> %{{[0-9]*}})

  /* vec_eqv */
  res_vsc =  vec_eqv(vsc, vsc);
// CHECK: [[T1:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <16 x i8>
// CHECK-PPC: error: assigning to

  res_vsc =  vec_eqv(vbc, vbc);
// CHECK: [[T1:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <16 x i8>
// CHECK-PPC: error: assigning to

  res_vuc =  vec_eqv(vuc, vuc);
// CHECK: [[T1:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <16 x i8> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <16 x i8>
// CHECK-PPC: error: assigning to

  res_vss =  vec_eqv(vss, vss);
// CHECK: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <8 x i16>
// CHECK-LE: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <8 x i16>
// CHECK-PPC: error: assigning to

  res_vss =  vec_eqv(vbs, vbs);
// CHECK: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <8 x i16>
// CHECK-LE: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <8 x i16>
// CHECK-PPC: error: assigning to

  res_vus =  vec_eqv(vus, vus);
// CHECK: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <8 x i16>
// CHECK-LE: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <8 x i16>
// CHECK-PPC: error: assigning to

  res_vsi =  vec_eqv(vsi, vsi);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> {{.*}}, <4 x i32> {{.+}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> {{.*}}, <4 x i32> {{.+}})
// CHECK-PPC: error: assigning to

  res_vsi =  vec_eqv(vbi, vbi);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> {{.*}}, <4 x i32> {{.+}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> {{.*}}, <4 x i32> {{.+}})
// CHECK-PPC: error: assigning to

  res_vui =  vec_eqv(vui, vui);
// CHECK: call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> {{.*}}, <4 x i32> {{.+}})
// CHECK-LE: call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> {{.*}}, <4 x i32> {{.+}})
// CHECK-PPC: error: assigning to

  res_vsll =  vec_eqv(vsll, vsll);
// CHECK: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <2 x i64>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <2 x i64>
// CHECK-PPC: error: assigning to

  res_vsll =  vec_eqv(vbll, vbll);
// CHECK: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <2 x i64>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <2 x i64>
// CHECK-PPC: error: assigning to

  res_vull =  vec_eqv(vull, vull);
// CHECK: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <2 x i64>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x i64> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <2 x i64>
// CHECK-PPC: error: assigning to

  res_vf = vec_eqv(vfa, vfa);
// CHECK: [[T1:%.+]] = bitcast <4 x float> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <4 x float> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <4 x float>
// CHECK-LE: [[T1:%.+]] = bitcast <4 x float> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <4 x float> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <4 x float>
// CHECK-PPC: error: assigning to

  res_vd = vec_eqv(vda, vda);
// CHECK: [[T1:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK: bitcast <4 x i32> [[T3]] to <2 x double>
// CHECK-LE: [[T1:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <2 x double> {{.+}} to <4 x i32>
// CHECK-LE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxleqv(<4 x i32> [[T1]], <4 x i32> [[T2]])
// CHECK-LE: bitcast <4 x i32> [[T3]] to <2 x double>
// CHECK-PPC: error: assigning to

  /* vec_extract */
  res_sll = vec_extract(vsll, si);
// CHECK: extractelement <2 x i64>
// CHECK-LE: extractelement <2 x i64>

  res_ull = vec_extract(vull, si);
// CHECK: extractelement <2 x i64>
// CHECK-LE: extractelement <2 x i64>

  res_ull = vec_extract(vbll, si);
// CHECK: extractelement <2 x i64>
// CHECK-LE: extractelement <2 x i64>

  res_d = vec_extract(vda, si);
// CHECK: extractelement <2 x double>
// CHECK-LE: extractelement <2 x double>

  /* vec_insert */
  res_vsll = vec_insert(sll, vsll, si);
// CHECK: insertelement <2 x i64>
// CHECK-LE: insertelement <2 x i64>

  res_vbll = vec_insert(ull, vbll, si);
// CHECK: insertelement <2 x i64>
// CHECK-LE: insertelement <2 x i64>

  res_vull = vec_insert(ull, vull, si);
// CHECK: insertelement <2 x i64>
// CHECK-LE: insertelement <2 x i64>

  res_vd = vec_insert(d, vda, si);
// CHECK: insertelement <2 x double>
// CHECK-LE: insertelement <2 x double>

  /* vec_cntlz */
  res_vsc = vec_cntlz(vsc);
// CHECK: call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %{{.+}}, i1 false)
// CHECK-LE: call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %{{.+}}, i1 false)
// CHECK-PPC: warning: implicit declaration of function 'vec_cntlz' is invalid in C99

  res_vuc = vec_cntlz(vuc);
// CHECK: call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %{{.+}}, i1 false)
// CHECK-LE: call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %{{.+}}, i1 false)

  res_vss = vec_cntlz(vss);
// CHECK: call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %{{.+}}, i1 false)
// CHECK-LE: call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %{{.+}}, i1 false)

  res_vus = vec_cntlz(vus);
// CHECK: call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %{{.+}}, i1 false)
// CHECK-LE: call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %{{.+}}, i1 false)

  res_vsi = vec_cntlz(vsi);
// CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.+}}, i1 false)
// CHECK-LE: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.+}}, i1 false)

  res_vui = vec_cntlz(vui);
// CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.+}}, i1 false)
// CHECK-LE: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.+}}, i1 false)

  res_vsll = vec_cntlz(vsll);
// CHECK: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.+}}, i1 false)
// CHECK-LE: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.+}}, i1 false)

  res_vull = vec_cntlz(vull);
// CHECK: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.+}}, i1 false)
// CHECK-LE: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.+}}, i1 false)

  /* ----------------------- predicates --------------------------- */
  res_i = vec_all_eq(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpeqdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqdp.p

  res_i = vec_all_eq(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpeqsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqsp.p

  dummy();
// CHECK: @dummy

  res_i = vec_all_ne(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpeqdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqdp.p

  dummy();
// CHECK: @dummy

  res_i = vec_all_ne(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpeqsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqsp.p

  dummy();
// CHECK: @dummy

  res_i = vec_all_nge(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgedp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgedp.p

  res_i = vec_all_ngt(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgtdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtdp.p

  res_i = vec_any_eq(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpeqdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqdp.p

  res_i = vec_any_eq(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpeqsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqsp.p

  res_i = vec_any_ne(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpeqdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqdp.p

  res_i = vec_any_ne(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpeqsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqsp.p

  res_i = vec_all_ge(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgedp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgedp.p

  res_i = vec_all_ge(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpgesp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgesp.p

  res_i = vec_all_gt(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgtdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtdp.p

  res_i = vec_all_gt(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpgtsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtsp.p

  res_i = vec_all_le(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgedp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgedp.p

  res_i = vec_all_le(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpgesp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgesp.p

  res_i = vec_all_lt(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgtdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtdp.p

  res_i = vec_all_lt(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpgtsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtsp.p

  res_i = vec_all_nan(vda);
// CHECK: @llvm.ppc.vsx.xvcmpeqdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqdp.p

  res_i = vec_all_nan(vfa);
// CHECK: @llvm.ppc.vsx.xvcmpeqsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpeqsp.p

  res_i = vec_any_ge(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgedp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgedp.p

  res_i = vec_any_ge(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpgesp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgesp.p

  res_i = vec_any_gt(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgtdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtdp.p

  res_i = vec_any_le(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgedp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgedp.p

  res_i = vec_any_le(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpgesp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgesp.p

  res_i = vec_any_lt(vda, vda);
// CHECK: @llvm.ppc.vsx.xvcmpgtdp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtdp.p

  res_i = vec_any_lt(vfa, vfa);
// CHECK: @llvm.ppc.vsx.xvcmpgtsp.p
// CHECK-LE: @llvm.ppc.vsx.xvcmpgtsp.p

  /* vec_max */
  res_vsll = vec_max(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vsll = vec_max(vbll, vsll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vsll = vec_max(vsll, vbll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vull = vec_max(vull, vull);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vull = vec_max(vbll, vull);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vull = vec_max(vull, vbll);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  /* vec_mergeh */
  res_vbll = vec_mergeh(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbll = vec_mergel(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_min */
  res_vsll = vec_min(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vsll = vec_min(vbll, vsll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vsll = vec_min(vsll, vbll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vull = vec_min(vull, vull);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vull = vec_min(vbll, vull);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vull = vec_min(vull, vbll);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  /* vec_mule */
  res_vsll = vec_mule(vsi, vsi);
// CHECK: @llvm.ppc.altivec.vmulesw
// CHECK-LE: @llvm.ppc.altivec.vmulosw
// CHECK-PPC: error: call to 'vec_mule' is ambiguous

  res_vull = vec_mule(vui , vui);
// CHECK: @llvm.ppc.altivec.vmuleuw
// CHECK-LE: @llvm.ppc.altivec.vmulouw
// CHECK-PPC: error: call to 'vec_mule' is ambiguous

  /* vec_mulo */
  res_vsll = vec_mulo(vsi, vsi);
// CHECK: @llvm.ppc.altivec.vmulosw
// CHECK-LE: @llvm.ppc.altivec.vmulesw
// CHECK-PPC: error: call to 'vec_mulo' is ambiguous

  res_vull = vec_mulo(vui, vui);
// CHECK: @llvm.ppc.altivec.vmulouw
// CHECK-LE: @llvm.ppc.altivec.vmuleuw
// CHECK-PPC: error: call to 'vec_mulo' is ambiguous

  /* vec_packs */
  res_vsi = vec_packs(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vpksdss
// CHECK-LE: @llvm.ppc.altivec.vpksdss
// CHECK-PPC: error: call to 'vec_packs' is ambiguous

  res_vui = vec_packs(vull, vull);
// CHECK: @llvm.ppc.altivec.vpkudus
// CHECK-LE: @llvm.ppc.altivec.vpkudus
// CHECK-PPC: error: call to 'vec_packs' is ambiguous

  /* vec_packsu */
  res_vui = vec_packsu(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vpksdus
// CHECK-LE: @llvm.ppc.altivec.vpksdus
// CHECK-PPC: error: call to 'vec_packsu' is ambiguous

  res_vui = vec_packsu(vull, vull);
// CHECK: @llvm.ppc.altivec.vpkudus
// CHECK-LE: @llvm.ppc.altivec.vpkudus
// CHECK-PPC: error: call to 'vec_packsu' is ambiguous

  /* vec_rl */
  res_vsll = vec_rl(vsll, vull);
// CHECK: @llvm.ppc.altivec.vrld
// CHECK-LE: @llvm.ppc.altivec.vrld

  res_vull = vec_rl(vull, vull);
// CHECK: @llvm.ppc.altivec.vrld
// CHECK-LE: @llvm.ppc.altivec.vrld

  /* vec_sl */
  res_vsll = vec_sl(vsll, vull);
// CHECK: shl <2 x i64>
// CHECK-LE: shl <2 x i64>

  res_vull = vec_sl(vull, vull);
// CHECK: shl <2 x i64>
// CHECK-LE: shl <2 x i64>

  /* vec_sr */
  res_vsll = vec_sr(vsll, vull);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <2 x i64> {{[0-9a-zA-Z%.]+}}, <i64 64, i64 64>
// CHECK: lshr <2 x i64> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <2 x i64> {{[0-9a-zA-Z%.]+}}, <i64 64, i64 64>
// CHECK-LE: lshr <2 x i64> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vull = vec_sr(vull, vull);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <2 x i64> {{[0-9a-zA-Z%.]+}}, <i64 64, i64 64>
// CHECK: lshr <2 x i64> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <2 x i64> {{[0-9a-zA-Z%.]+}}, <i64 64, i64 64>
// CHECK-LE: lshr <2 x i64> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  /* vec_sra */
  res_vsll = vec_sra(vsll, vull);
// CHECK: ashr <2 x i64>
// CHECK-LE: ashr <2 x i64>

  res_vull = vec_sra(vull, vull);
// CHECK: ashr <2 x i64>
// CHECK-LE: ashr <2 x i64>

  /* vec_splats */
  res_vsll = vec_splats(sll);
// CHECK: insertelement <2 x i64>
// CHECK-LE: insertelement <2 x i64>

  res_vull = vec_splats(ull);
// CHECK: insertelement <2 x i64>
// CHECK-LE: insertelement <2 x i64>

  res_vsx = vec_splats(sx);
// CHECK: insertelement <1 x i128>
// CHECK-LE: insertelement <1 x i128>

  res_vux = vec_splats(ux);
// CHECK: insertelement <1 x i128>
// CHECK-LE: insertelement <1 x i128>

  res_vd = vec_splats(d);
// CHECK: insertelement <2 x double>
// CHECK-LE: insertelement <2 x double>


  /* vec_unpackh */
  res_vsll = vec_unpackh(vsi);
// CHECK: llvm.ppc.altivec.vupkhsw
// CHECK-LE: llvm.ppc.altivec.vupklsw
// CHECK-PPC: error: call to 'vec_unpackh' is ambiguous

  res_vbll = vec_unpackh(vbi);
// CHECK: llvm.ppc.altivec.vupkhsw
// CHECK-LE: llvm.ppc.altivec.vupklsw
// CHECK-PPC: error: call to 'vec_unpackh' is ambiguous

  /* vec_unpackl */
  res_vsll = vec_unpackl(vsi);
// CHECK: llvm.ppc.altivec.vupklsw
// CHECK-LE: llvm.ppc.altivec.vupkhsw
// CHECK-PPC: error: call to 'vec_unpackl' is ambiguous

  res_vbll = vec_unpackl(vbi);
// CHECK: llvm.ppc.altivec.vupklsw
// CHECK-LE: llvm.ppc.altivec.vupkhsw
// CHECK-PPC: error: call to 'vec_unpackl' is ambiguous

  /* vec_vpksdss */
  res_vsi = vec_vpksdss(vsll, vsll);
// CHECK: llvm.ppc.altivec.vpksdss
// CHECK-LE: llvm.ppc.altivec.vpksdss
// CHECK-PPC: warning: implicit declaration of function 'vec_vpksdss'

  /* vec_vpksdus */
  res_vui = vec_vpksdus(vsll, vsll);
// CHECK: llvm.ppc.altivec.vpksdus
// CHECK-LE: llvm.ppc.altivec.vpksdus
// CHECK-PPC: warning: implicit declaration of function 'vec_vpksdus'

  /* vec_vpkudum */
  res_vsi = vec_vpkudum(vsll, vsll);
// CHECK: vperm
// CHECK-LE: vperm
// CHECK-PPC: warning: implicit declaration of function 'vec_vpkudum'

  res_vui = vec_vpkudum(vull, vull);
// CHECK: vperm
// CHECK-LE: vperm

  res_vui = vec_vpkudus(vull, vull);
// CHECK: llvm.ppc.altivec.vpkudus
// CHECK-LE: llvm.ppc.altivec.vpkudus
// CHECK-PPC: warning: implicit declaration of function 'vec_vpkudus'

  /* vec_vupkhsw */
  res_vsll = vec_vupkhsw(vsi);
// CHECK: llvm.ppc.altivec.vupkhsw
// CHECK-LE: llvm.ppc.altivec.vupklsw
// CHECK-PPC: warning: implicit declaration of function 'vec_vupkhsw'

  res_vbll = vec_vupkhsw(vbi);
// CHECK: llvm.ppc.altivec.vupkhsw
// CHECK-LE: llvm.ppc.altivec.vupklsw

  /* vec_vupklsw */
  res_vsll = vec_vupklsw(vsi);
// CHECK: llvm.ppc.altivec.vupklsw
// CHECK-LE: llvm.ppc.altivec.vupkhsw
// CHECK-PPC: warning: implicit declaration of function 'vec_vupklsw'

  res_vbll = vec_vupklsw(vbi);
// CHECK: llvm.ppc.altivec.vupklsw
// CHECK-LE: llvm.ppc.altivec.vupkhsw

  /* vec_max */
  res_vsll = vec_max(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd

  res_vsll = vec_max(vbll, vsll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd

  res_vsll = vec_max(vsll, vbll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd

  res_vull = vec_max(vull, vull);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud

  res_vull = vec_max(vbll, vull);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud

  /* vec_min */
  res_vsll = vec_min(vsll, vsll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd

  res_vsll = vec_min(vbll, vsll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd

  res_vsll = vec_min(vsll, vbll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd

  res_vull = vec_min(vull, vull);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud

  res_vull = vec_min(vbll, vull);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud

  /* vec_nand */
  res_vsc = vec_nand(vsc, vsc);
// CHECK: [[T1:%.+]] = and <16 x i8>
// CHECK: xor <16 x i8> [[T1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: [[T1:%.+]] = and <16 x i8>
// CHECK-LE: xor <16 x i8> [[T1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-PPC: warning: implicit declaration of function 'vec_nand' is invalid in C99

  res_vsc = vec_nand(vbc, vbc);
// CHECK: [[T1:%.+]] = and <16 x i8>
// CHECK: xor <16 x i8> [[T1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: [[T1:%.+]] = and <16 x i8>
// CHECK-LE: xor <16 x i8> [[T1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  
  res_vuc = vec_nand(vuc, vuc);
// CHECK: [[T1:%.+]] = and <16 x i8>
// CHECK: xor <16 x i8> [[T1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: [[T1:%.+]] = and <16 x i8>
// CHECK-LE: xor <16 x i8> [[T1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  
  res_vss = vec_nand(vss, vss);
// CHECK: [[T1:%.+]] = and <8 x i16>
// CHECK: xor <8 x i16> [[T1]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: [[T1:%.+]] = and <8 x i16>
// CHECK-LE: xor <8 x i16> [[T1]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>

  res_vss = vec_nand(vbs, vbs);
// CHECK: [[T1:%.+]] = and <8 x i16>
// CHECK: xor <8 x i16> [[T1]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: [[T1:%.+]] = and <8 x i16>
// CHECK-LE: xor <8 x i16> [[T1]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>

  res_vus = vec_nand(vus, vus);
// CHECK: [[T1:%.+]] = and <8 x i16>
// CHECK: xor <8 x i16> [[T1]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: [[T1:%.+]] = and <8 x i16>
// CHECK-LE: xor <8 x i16> [[T1]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>

  res_vsi = vec_nand(vsi, vsi);
// CHECK: [[T1:%.+]] = and <4 x i32>
// CHECK: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: [[T1:%.+]] = and <4 x i32>
// CHECK-LE: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>

  res_vsi = vec_nand(vbi, vbi);
// CHECK: [[T1:%.+]] = and <4 x i32>
// CHECK: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: [[T1:%.+]] = and <4 x i32>
// CHECK-LE: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>

  res_vui = vec_nand(vui, vui);
// CHECK: [[T1:%.+]] = and <4 x i32>
// CHECK: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: [[T1:%.+]] = and <4 x i32>
// CHECK-LE: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>

  res_vf = vec_nand(vfa, vfa);
// CHECK: [[T1:%.+]] = and <4 x i32>
// CHECK: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: [[T1:%.+]] = and <4 x i32>
// CHECK-LE: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>

  res_vsll = vec_nand(vsll, vsll);
// CHECK: [[T1:%.+]] = and <2 x i64>
// CHECK: xor <2 x i64> [[T1]], <i64 -1, i64 -1>
// CHECK-LE: [[T1:%.+]] = and <2 x i64>
// CHECK-LE: xor <2 x i64> [[T1]], <i64 -1, i64 -1>

  res_vsll = vec_nand(vbll, vbll);
// CHECK: [[T1:%.+]] = and <2 x i64>
// CHECK: xor <2 x i64> [[T1]], <i64 -1, i64 -1>
// CHECK-LE: [[T1:%.+]] = and <2 x i64>
// CHECK-LE: xor <2 x i64> [[T1]], <i64 -1, i64 -1>

  res_vull = vec_nand(vull, vull);
// CHECK: [[T1:%.+]] = and <2 x i64>
// CHECK: xor <2 x i64> [[T1]], <i64 -1, i64 -1>
// CHECK-LE: [[T1:%.+]] = and <2 x i64>
// CHECK-LE: xor <2 x i64> [[T1]], <i64 -1, i64 -1>

  res_vd = vec_nand(vda, vda);
// CHECK: [[T1:%.+]] = and <2 x i64>
// CHECK: xor <2 x i64> [[T1]], <i64 -1, i64 -1>
// CHECK-LE: [[T1:%.+]] = and <2 x i64>
// CHECK-LE: xor <2 x i64> [[T1]], <i64 -1, i64 -1>

  res_vf = vec_nand(vfa, vfa);
// CHECK: [[T1:%.+]] = and <4 x i32>
// CHECK: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: [[T1:%.+]] = and <4 x i32>
// CHECK-LE: xor <4 x i32> [[T1]], <i32 -1, i32 -1, i32 -1, i32 -1>

  /* vec_orc */
  res_vsc = vec_orc(vsc, vsc);
// CHECK: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-PPC: warning: implicit declaration of function 'vec_orc' is invalid in C99

  res_vsc = vec_orc(vsc, vbc);
// CHECK: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: or <16 x i8> {{%.+}}, [[T1]]

  res_vsc = vec_orc(vbc, vsc);
// CHECK: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: or <16 x i8> {{%.+}}, [[T1]]

  res_vuc = vec_orc(vuc, vuc);
// CHECK: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: or <16 x i8> {{%.+}}, [[T1]]

  res_vuc = vec_orc(vuc, vbc);
// CHECK: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: or <16 x i8> {{%.+}}, [[T1]]

  res_vuc = vec_orc(vbc, vuc);
// CHECK: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: or <16 x i8> {{%.+}}, [[T1]]

  res_vbc = vec_orc(vbc, vbc);
// CHECK: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK: or <16 x i8> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <16 x i8> {{%.+}}, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK-LE: or <16 x i8> {{%.+}}, [[T1]]

  res_vss = vec_orc(vss, vss);
// CHECK: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK: or <8 x i16> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: or <8 x i16> {{%.+}}, [[T1]]

  res_vss = vec_orc(vss, vbs);
// CHECK: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK: or <8 x i16> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: or <8 x i16> {{%.+}}, [[T1]]

  res_vss = vec_orc(vbs, vss);
// CHECK: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK: or <8 x i16> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: or <8 x i16> {{%.+}}, [[T1]]

  res_vus = vec_orc(vus, vus);
// CHECK: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK: or <8 x i16> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: or <8 x i16> {{%.+}}, [[T1]]

  res_vus = vec_orc(vus, vbs);
// CHECK: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK: or <8 x i16> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: or <8 x i16> {{%.+}}, [[T1]]

  res_vus = vec_orc(vbs, vus);
// CHECK: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK: or <8 x i16> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: or <8 x i16> {{%.+}}, [[T1]]

  res_vbs = vec_orc(vbs, vbs);
// CHECK: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK: or <8 x i16> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <8 x i16> {{%.+}}, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK-LE: or <8 x i16> {{%.+}}, [[T1]]

  res_vsi = vec_orc(vsi, vsi);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vsi = vec_orc(vsi, vbi);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vsi = vec_orc(vbi, vsi);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vui = vec_orc(vui, vui);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vui = vec_orc(vui, vbi);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vui = vec_orc(vbi, vui);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vbi = vec_orc(vbi, vbi);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vf = vec_orc(vbi, vfa);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vf = vec_orc(vfa, vbi);
// CHECK: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: or <4 x i32> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <4 x i32> {{%.+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK-LE: or <4 x i32> {{%.+}}, [[T1]]

  res_vsll = vec_orc(vsll, vsll);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vsll = vec_orc(vsll, vbll);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vsll = vec_orc(vbll, vsll);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vull = vec_orc(vull, vull);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vull = vec_orc(vull, vbll);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vull = vec_orc(vbll, vull);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vbll = vec_orc(vbll, vbll);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vd = vec_orc(vbll, vda);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  res_vd = vec_orc(vda, vbll);
// CHECK: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK: or <2 x i64> {{%.+}}, [[T1]]
// CHECK-LE: [[T1:%.+]] = xor <2 x i64> {{%.+}}, <i64 -1, i64 -1>
// CHECK-LE: or <2 x i64> {{%.+}}, [[T1]]

  /* vec_sub */
  res_vsll = vec_sub(vsll, vsll);
// CHECK: sub <2 x i64>
// CHECK-LE: sub <2 x i64>

  res_vull = vec_sub(vull, vull);
// CHECK: sub <2 x i64>
// CHECK-LE: sub <2 x i64>

  res_vd = vec_sub(vda, vda);
// CHECK: fsub <2 x double>
// CHECK-LE: fsub <2 x double>

  res_vsx = vec_sub(vsx, vsx);
// CHECK: sub <1 x i128>
// CHECK-LE: sub <1 x i128>

  res_vux = vec_sub(vux, vux);
// CHECK: sub <1 x i128>
// CHECK-LE: sub <1 x i128>

  /* vec_vbpermq */
  res_vsll = vec_vbpermq(vsc, vsc);
// CHECK: llvm.ppc.altivec.vbpermq
// CHECK-LE: llvm.ppc.altivec.vbpermq

  res_vull = vec_vbpermq(vuc, vuc);
// CHECK: llvm.ppc.altivec.vbpermq
// CHECK-LE: llvm.ppc.altivec.vbpermq
// CHECK-PPC: warning: implicit declaration of function 'vec_vbpermq'

  /* vec_vgbbd */
  res_vsc = vec_vgbbd(vsc);
// CHECK: llvm.ppc.altivec.vgbbd
// CHECK-LE: llvm.ppc.altivec.vgbbd

  res_vuc = vec_vgbbd(vuc);
// CHECK: llvm.ppc.altivec.vgbbd
// CHECK-LE: llvm.ppc.altivec.vgbbd
// CHECK-PPC: warning: implicit declaration of function 'vec_vgbbd'

  res_vuc = vec_gb(vuc);
// CHECK: llvm.ppc.altivec.vgbbd
// CHECK-LE: llvm.ppc.altivec.vgbbd
// CHECK-PPC: warning: implicit declaration of function 'vec_gb'

  res_vsll = vec_gbb(vsll);
// CHECK: llvm.ppc.altivec.vgbbd
// CHECK-LE: llvm.ppc.altivec.vgbbd

  res_vull = vec_gbb(vull);
// CHECK: llvm.ppc.altivec.vgbbd
// CHECK-LE: llvm.ppc.altivec.vgbbd

  res_vull = vec_bperm(vux, vux);
// CHECK: llvm.ppc.altivec.vbpermq
// CHECK-LE: llvm.ppc.altivec.vbpermq
// CHECK-PPC: warning: implicit declaration of function 'vec_bperm'

  res_vsll = vec_neg(vsll);
// CHECK: sub <2 x i64> zeroinitializer, {{%[0-9]+}}
// CHECK-LE: sub <2 x i64> zeroinitializer, {{%[0-9]+}}
// CHECK_PPC: call to 'vec_neg' is ambiguous


}


vector signed int test_vec_addec_signed (vector signed int a, vector signed int b, vector signed int c) {
  return vec_addec(a, b, c);
// CHECK-LABEL: @test_vec_addec_signed
// CHECK: icmp slt i32 {{%[0-9]+}}, 4
// CHECK: extractelement
// CHECK: extractelement
// CHECK: extractelement
// CHECK: and i32 {{%[0-9]+}}, 1
// CHECK: zext
// CHECK: zext
// CHECK: zext
// CHECK: add i64
// CHECK: add i64
// CHECK: lshr i64
// CHECK: and i64
// CHECK: trunc i64 {{%[0-9]+}} to i32
// CHECK: zext i32
// CHECK: trunc i64 {{%[0-9]+}} to i32
// CHECK: sext i32
// CHECK: add nsw i32
// CHECK: br label
// CHECK: ret <4 x i32>

}


vector unsigned int test_vec_addec_unsigned (vector unsigned int a, vector unsigned int b, vector unsigned int c) {
  return vec_addec(a, b, c);

// CHECK-LABEL: @test_vec_addec_unsigned
// CHECK: icmp slt i32 {{%[0-9]+}}, 4
// CHECK: extractelement
// CHECK: and i32
// CHECK: extractelement
// CHECK: zext i32
// CHECK: extractelement
// CHECK: zext i32
// CHECK: zext i32
// CHECK: add i64
// CHECK: lshr i64
// CHECK: and i64
// CHECK: trunc i64 {{%[0-9]+}} to i32
// CHECK: zext i32
// CHECK: trunc i64 {{%[0-9]+}} to i32
// CHECK: sext i32
// CHECK: add nsw i32
// CHECK: br label
// CHECK: ret <4 x i32>
}

vector signed int test_vec_subec_signed (vector signed int a, vector signed int b, vector signed int c) {
  return vec_subec(a, b, c);
// CHECK-LABEL: @test_vec_subec_signed
// CHECK: xor <4 x i32> {{%[0-9]+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: ret <4 x i32>
}

vector unsigned int test_vec_subec_unsigned (vector unsigned int a, vector unsigned int b, vector unsigned int c) {
  return vec_subec(a, b, c);

// CHECK-LABEL: @test_vec_subec_unsigned
// CHECK: xor <4 x i32> {{%[0-9]+}}, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK: ret <4 x i32>
}
