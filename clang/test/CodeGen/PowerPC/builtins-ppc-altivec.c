// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc-unknown-unknown -emit-llvm %s \
// RUN:            -flax-vector-conversions=none  -faltivec-src-compat=mixed \
// RUN:            -o - | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -triple powerpcle-unknown-unknown -emit-llvm %s \
// RUN:            -flax-vector-conversions=none  -faltivec-src-compat=mixed \
// RUN:            -o - | FileCheck %s -check-prefix=CHECK-LE
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:            -flax-vector-conversions=none  -faltivec-src-compat=mixed \
// RUN:            -o - | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:            -flax-vector-conversions=none  -faltivec-src-compat=mixed \
// RUN:            -o - | FileCheck %s -check-prefix=CHECK-LE
// RUN: not %clang_cc1 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:            -ferror-limit 0 -DNO_ALTIVEC -o - 2>&1 \
// RUN:            | FileCheck %s -check-prefix=CHECK-NOALTIVEC
#ifndef NO_ALTIVEC
#include <altivec.h>
#endif

vector bool char vbc = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
vector signed char vsc = { 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16 };
vector unsigned char vuc = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
vector bool short vbs = { 1, 0, 1, 0, 1, 0, 1, 0 };
vector short vs = { -1, 2, -3, 4, -5, 6, -7, 8 };
vector unsigned short vus = { 1, 2, 3, 4, 5, 6, 7, 8 };
vector pixel vp = { 1, 2, 3, 4, 5, 6, 7, 8 };
vector bool int vbi = { 1, 0, 1, 0 };
vector int vi = { -1, 2, -3, 4 };
vector unsigned int vui = { 1, 2, 3, 4 };
vector float vf = { -1.5, 2.5, -3.5, 4.5 };

vector bool char res_vbc;
vector signed char res_vsc;
vector unsigned char res_vuc;
vector bool short res_vbs;
vector short res_vs;
vector unsigned short res_vus;
vector pixel res_vp;
vector bool int res_vbi;
vector int res_vi;
vector unsigned int res_vui;
vector float res_vf;

// CHECK-NOALTIVEC: error: unknown type name 'vector'
// CHECK-NOALTIVEC-NOT: '(error)'
const signed char *param_sc_ld;
const unsigned char *param_uc_ld;
const short *param_s_ld;
const unsigned short *param_us_ld;
const int *param_i_ld;
const unsigned int *param_ui_ld;
const float *param_f_ld;

signed char param_sc;
unsigned char param_uc;
short param_s;
unsigned short param_us;
int param_i;
unsigned int param_ui;
float param_f;
signed long long param_sll;

int res_sc;
int res_uc;
int res_s;
int res_us;
int res_i;
int res_ui;
int res_f;

// CHECK-LABEL: define{{.*}} void @test1
void test1() {

  /* vec_abs */
  vsc = vec_abs(vsc);
// CHECK: sub <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: sub <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  vs = vec_abs(vs);
// CHECK: sub <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: sub <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  vi = vec_abs(vi);
// CHECK: sub <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: sub <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  vf = vec_abs(vf);
// CHECK: and <4 x i32> {{.*}}, splat (i32 2147483647)
// CHECK: store <4 x float> %{{.*}}, ptr @vf
// CHECK-LE: bitcast <4 x float> %{{.*}} to <4 x i32>
// CHECK-LE: and <4 x i32> {{.*}}, splat (i32 2147483647)
// CHECK-LE: bitcast <4 x i32> %{{.*}} to <4 x float>
// CHECK-LE: store <4 x float> %{{.*}}, ptr @vf

  vsc = vec_nabs(vsc);
// CHECK: sub <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vminsb
// CHECK-LE: sub <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vminsb

  vs = vec_nabs(vs);
// CHECK: sub <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vminsh
// CHECK-LE: sub <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vminsh

  vi = vec_nabs(vi);
// CHECK: sub <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vminsw
// CHECK-LE: sub <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vminsw

  res_vi = vec_neg(vi);
// CHECK: sub <4 x i32> zeroinitializer, {{%[0-9]+}}
// CHECK-LE: sub <4 x i32> zeroinitializer, {{%[0-9]+}}

  res_vs = vec_neg(vs);
// CHECK: sub <8 x i16> zeroinitializer, {{%[0-9]+}}
// CHECK-LE: sub <8 x i16> zeroinitializer, {{%[0-9]+}}

  res_vsc = vec_neg(vsc);
// CHECK: sub <16 x i8> zeroinitializer, {{%[0-9]+}}
// CHECK-LE: sub <16 x i8> zeroinitializer, {{%[0-9]+}}


  /* vec_abs */
  vsc = vec_abss(vsc);
// CHECK: @llvm.ppc.altivec.vsubsbs
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: @llvm.ppc.altivec.vsubsbs
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  vs = vec_abss(vs);
// CHECK: @llvm.ppc.altivec.vsubshs
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: @llvm.ppc.altivec.vsubshs
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  vi = vec_abss(vi);
// CHECK: @llvm.ppc.altivec.vsubsws
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: @llvm.ppc.altivec.vsubsws
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  /*  vec_add */
  res_vsc = vec_add(vsc, vsc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vsc = vec_add(vbc, vsc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vsc = vec_add(vsc, vbc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vuc = vec_add(vuc, vuc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vuc = vec_add(vbc, vuc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vuc = vec_add(vuc, vbc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vs  = vec_add(vs, vs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vs  = vec_add(vbs, vs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vs  = vec_add(vs, vbs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vus = vec_add(vus, vus);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vus = vec_add(vbs, vus);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vus = vec_add(vus, vbs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vi  = vec_add(vi, vi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vi  = vec_add(vbi, vi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vi  = vec_add(vi, vbi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui = vec_add(vui, vui);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui = vec_add(vbi, vui);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui = vec_add(vui, vbi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vf  = vec_add(vf, vf);
// CHECK: fadd <4 x float>
// CHECK-LE: fadd <4 x float>

  res_vi  = vec_adde(vi, vi, vi);
// CHECK: and <4 x i32>
// CHECK: add <4 x i32>
// CHECK: add <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui  = vec_adde(vui, vui, vui);
// CHECK: and <4 x i32>
// CHECK: add <4 x i32>
// CHECK: add <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vsc = vec_vaddubm(vsc, vsc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vsc = vec_vaddubm(vbc, vsc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vsc = vec_vaddubm(vsc, vbc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vuc = vec_vaddubm(vuc, vuc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vuc = vec_vaddubm(vbc, vuc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vuc = vec_vaddubm(vuc, vbc);
// CHECK: add <16 x i8>
// CHECK-LE: add <16 x i8>

  res_vs  = vec_vadduhm(vs, vs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vs  = vec_vadduhm(vbs, vs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vs  = vec_vadduhm(vs, vbs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vus = vec_vadduhm(vus, vus);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vus = vec_vadduhm(vbs, vus);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vus = vec_vadduhm(vus, vbs);
// CHECK: add <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vi  = vec_vadduwm(vi, vi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vi  = vec_vadduwm(vbi, vi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vi  = vec_vadduwm(vi, vbi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui = vec_vadduwm(vui, vui);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui = vec_vadduwm(vbi, vui);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui = vec_vadduwm(vui, vbi);
// CHECK: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vf  = vec_vaddfp(vf, vf);
// CHECK: fadd <4 x float>
// CHECK-LE: fadd <4 x float>

  /* vec_addc */
  res_vui = vec_addc(vui, vui);
// CHECK: @llvm.ppc.altivec.vaddcuw
// CHECK-LE: @llvm.ppc.altivec.vaddcuw

  res_vui = vec_vaddcuw(vui, vui);
// CHECK: @llvm.ppc.altivec.vaddcuw
// CHECK-LE: @llvm.ppc.altivec.vaddcuw

  /* vec_adds */
  res_vsc = vec_adds(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vaddsbs
// CHECK-LE: @llvm.ppc.altivec.vaddsbs

  res_vsc = vec_adds(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vaddsbs
// CHECK-LE: @llvm.ppc.altivec.vaddsbs

  res_vsc = vec_adds(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vaddsbs
// CHECK-LE: @llvm.ppc.altivec.vaddsbs

  res_vuc = vec_adds(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vaddubs
// CHECK-LE: @llvm.ppc.altivec.vaddubs

  res_vuc = vec_adds(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vaddubs
// CHECK-LE: @llvm.ppc.altivec.vaddubs

  res_vuc = vec_adds(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vaddubs
// CHECK-LE: @llvm.ppc.altivec.vaddubs

  res_vs  = vec_adds(vs, vs);
// CHECK: @llvm.ppc.altivec.vaddshs
// CHECK-LE: @llvm.ppc.altivec.vaddshs

  res_vs  = vec_adds(vbs, vs);
// CHECK: @llvm.ppc.altivec.vaddshs
// CHECK-LE: @llvm.ppc.altivec.vaddshs

  res_vs  = vec_adds(vs, vbs);
// CHECK: @llvm.ppc.altivec.vaddshs
// CHECK-LE: @llvm.ppc.altivec.vaddshs

  res_vus = vec_adds(vus, vus);
// CHECK: @llvm.ppc.altivec.vadduhs
// CHECK-LE: @llvm.ppc.altivec.vadduhs

  res_vus = vec_adds(vbs, vus);
// CHECK: @llvm.ppc.altivec.vadduhs
// CHECK-LE: @llvm.ppc.altivec.vadduhs

  res_vus = vec_adds(vus, vbs);
// CHECK: @llvm.ppc.altivec.vadduhs
// CHECK-LE: @llvm.ppc.altivec.vadduhs

  res_vi  = vec_adds(vi, vi);
// CHECK: @llvm.ppc.altivec.vaddsws
// CHECK-LE: @llvm.ppc.altivec.vaddsws

  res_vi  = vec_adds(vbi, vi);
// CHECK: @llvm.ppc.altivec.vaddsws
// CHECK-LE: @llvm.ppc.altivec.vaddsws

  res_vi  = vec_adds(vi, vbi);
// CHECK: @llvm.ppc.altivec.vaddsws
// CHECK-LE: @llvm.ppc.altivec.vaddsws

  res_vui = vec_adds(vui, vui);
// CHECK: @llvm.ppc.altivec.vadduws
// CHECK-LE: @llvm.ppc.altivec.vadduws

  res_vui = vec_adds(vbi, vui);
// CHECK: @llvm.ppc.altivec.vadduws
// CHECK-LE: @llvm.ppc.altivec.vadduws

  res_vui = vec_adds(vui, vbi);
// CHECK: @llvm.ppc.altivec.vadduws
// CHECK-LE: @llvm.ppc.altivec.vadduws

  res_vsc = vec_vaddsbs(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vaddsbs
// CHECK-LE: @llvm.ppc.altivec.vaddsbs

  res_vsc = vec_vaddsbs(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vaddsbs
// CHECK-LE: @llvm.ppc.altivec.vaddsbs

  res_vsc = vec_vaddsbs(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vaddsbs
// CHECK-LE: @llvm.ppc.altivec.vaddsbs

  res_vuc = vec_vaddubs(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vaddubs
// CHECK-LE: @llvm.ppc.altivec.vaddubs

  res_vuc = vec_vaddubs(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vaddubs
// CHECK-LE: @llvm.ppc.altivec.vaddubs

  res_vuc = vec_vaddubs(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vaddubs
// CHECK-LE: @llvm.ppc.altivec.vaddubs

  res_vs  = vec_vaddshs(vs, vs);
// CHECK: @llvm.ppc.altivec.vaddshs
// CHECK-LE: @llvm.ppc.altivec.vaddshs

  res_vs  = vec_vaddshs(vbs, vs);
// CHECK: @llvm.ppc.altivec.vaddshs
// CHECK-LE: @llvm.ppc.altivec.vaddshs

  res_vs  = vec_vaddshs(vs, vbs);
// CHECK: @llvm.ppc.altivec.vaddshs
// CHECK-LE: @llvm.ppc.altivec.vaddshs

  res_vus = vec_vadduhs(vus, vus);
// CHECK: @llvm.ppc.altivec.vadduhs
// CHECK-LE: @llvm.ppc.altivec.vadduhs

  res_vus = vec_vadduhs(vbs, vus);
// CHECK: @llvm.ppc.altivec.vadduhs
// CHECK-LE: @llvm.ppc.altivec.vadduhs

  res_vus = vec_vadduhs(vus, vbs);
// CHECK: @llvm.ppc.altivec.vadduhs
// CHECK-LE: @llvm.ppc.altivec.vadduhs

  res_vi  = vec_vaddsws(vi, vi);
// CHECK: @llvm.ppc.altivec.vaddsws
// CHECK-LE: @llvm.ppc.altivec.vaddsws

  res_vi  = vec_vaddsws(vbi, vi);
// CHECK: @llvm.ppc.altivec.vaddsws
// CHECK-LE: @llvm.ppc.altivec.vaddsws

  res_vi  = vec_vaddsws(vi, vbi);
// CHECK: @llvm.ppc.altivec.vaddsws
// CHECK-LE: @llvm.ppc.altivec.vaddsws

  res_vui = vec_vadduws(vui, vui);
// CHECK: @llvm.ppc.altivec.vadduws
// CHECK-LE: @llvm.ppc.altivec.vadduws

  res_vui = vec_vadduws(vbi, vui);
// CHECK: @llvm.ppc.altivec.vadduws
// CHECK-LE: @llvm.ppc.altivec.vadduws

  res_vui = vec_vadduws(vui, vbi);
// CHECK: @llvm.ppc.altivec.vadduws
// CHECK-LE: @llvm.ppc.altivec.vadduws

  /* vec_and */
  res_vsc = vec_and(vsc, vsc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_and(vbc, vsc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_and(vsc, vbc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_and(vuc, vuc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_and(vbc, vuc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_and(vuc, vbc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vbc = vec_and(vbc, vbc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vs  = vec_and(vs, vs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_and(vbs, vs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_and(vs, vbs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_and(vus, vus);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_and(vbs, vus);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_and(vus, vbs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vbs = vec_and(vbs, vbs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vi  = vec_and(vi, vi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vi  = vec_and(vbi, vi);
// CHECK: and <4 x i32>
// CHECK-le: and <4 x i32>

  res_vi  = vec_and(vi, vbi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_and(vui, vui);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_and(vbi, vui);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_and(vui, vbi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vbi = vec_and(vbi, vbi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vsc = vec_vand(vsc, vsc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_vand(vbc, vsc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_vand(vsc, vbc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_vand(vuc, vuc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_vand(vbc, vuc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_vand(vuc, vbc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vbc = vec_vand(vbc, vbc);
// CHECK: and <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vs  = vec_vand(vs, vs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_vand(vbs, vs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_vand(vs, vbs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_vand(vus, vus);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_vand(vbs, vus);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_vand(vus, vbs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vbs = vec_vand(vbs, vbs);
// CHECK: and <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vi  = vec_vand(vi, vi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vi  = vec_vand(vbi, vi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vi  = vec_vand(vi, vbi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_vand(vui, vui);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_vand(vbi, vui);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_vand(vui, vbi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vbi = vec_vand(vbi, vbi);
// CHECK: and <4 x i32>
// CHECK-LE: and <4 x i32>

  /* vec_andc */
  res_vsc = vec_andc(vsc, vsc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_andc(vbc, vsc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_andc(vsc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_andc(vuc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_andc(vbc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_andc(vuc, vbc);
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vbc = vec_andc(vbc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vs  = vec_andc(vs, vs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_andc(vbs, vs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_andc(vs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_andc(vus, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_andc(vbs, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_andc(vus, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vbs = vec_andc(vbs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vi  = vec_andc(vi, vi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vi  = vec_andc(vbi, vi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vi  = vec_andc(vi, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_andc(vui, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_andc(vbi, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_andc(vui, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vf = vec_andc(vf, vf);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vf = vec_andc(vbi, vf);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vf = vec_andc(vf, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vsc = vec_vandc(vsc, vsc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_vandc(vbc, vsc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vsc = vec_vandc(vsc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_vandc(vuc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_vandc(vbc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vuc = vec_vandc(vuc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vbc = vec_vandc(vbc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>

  res_vs  = vec_vandc(vs, vs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_vandc(vbs, vs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vs  = vec_vandc(vs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_vandc(vus, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_vandc(vbs, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vus = vec_vandc(vus, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vbs = vec_vandc(vbs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>

  res_vi  = vec_vandc(vi, vi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vi  = vec_vandc(vbi, vi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vi  = vec_vandc(vi, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_vandc(vui, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_vandc(vbi, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vui = vec_vandc(vui, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vf = vec_vandc(vf, vf);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vf = vec_vandc(vbi, vf);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

  res_vf = vec_vandc(vf, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>

}

// CHECK-LABEL: define{{.*}} void @test2
void test2() {
  /* vec_avg */
  res_vsc = vec_avg(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vavgsb
// CHECK-LE: @llvm.ppc.altivec.vavgsb

  res_vuc = vec_avg(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vavgub
// CHECK-LE: @llvm.ppc.altivec.vavgub

  res_vs  = vec_avg(vs, vs);
// CHECK: @llvm.ppc.altivec.vavgsh
// CHECK-LE: @llvm.ppc.altivec.vavgsh

  res_vus = vec_avg(vus, vus);
// CHECK: @llvm.ppc.altivec.vavguh
// CHECK-LE: @llvm.ppc.altivec.vavguh

  res_vi  = vec_avg(vi, vi);
// CHECK: @llvm.ppc.altivec.vavgsw
// CHECK-LE: @llvm.ppc.altivec.vavgsw

  res_vui = vec_avg(vui, vui);
// CHECK: @llvm.ppc.altivec.vavguw
// CHECK-LE: @llvm.ppc.altivec.vavguw

  res_vsc = vec_vavgsb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vavgsb
// CHECK-LE: @llvm.ppc.altivec.vavgsb

  res_vuc = vec_vavgub(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vavgub
// CHECK-LE: @llvm.ppc.altivec.vavgub

  res_vs  = vec_vavgsh(vs, vs);
// CHECK: @llvm.ppc.altivec.vavgsh
// CHECK-LE: @llvm.ppc.altivec.vavgsh

  res_vus = vec_vavguh(vus, vus);
// CHECK: @llvm.ppc.altivec.vavguh
// CHECK-LE: @llvm.ppc.altivec.vavguh

  res_vi  = vec_vavgsw(vi, vi);
// CHECK: @llvm.ppc.altivec.vavgsw
// CHECK-LE: @llvm.ppc.altivec.vavgsw

  res_vui = vec_vavguw(vui, vui);
// CHECK: @llvm.ppc.altivec.vavguw
// CHECK-LE: @llvm.ppc.altivec.vavguw

  /* vec_ceil */
  res_vf = vec_ceil(vf);
// CHECK: @llvm.ppc.altivec.vrfip
// CHECK-LE: @llvm.ppc.altivec.vrfip

  res_vf = vec_vrfip(vf);
// CHECK: @llvm.ppc.altivec.vrfip
// CHECK-LE: @llvm.ppc.altivec.vrfip

  /* vec_cmpb */
  res_vi = vec_cmpb(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpbfp
// CHECK-LE: @llvm.ppc.altivec.vcmpbfp

  res_vi = vec_vcmpbfp(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpbfp
// CHECK-LE: @llvm.ppc.altivec.vcmpbfp

  /* vec_cmpeq */
  res_vbc = vec_cmpeq(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: @llvm.ppc.altivec.vcmpequb

  res_vbc = vec_cmpeq(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: @llvm.ppc.altivec.vcmpequb

  res_vbc = vec_cmpeq(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: @llvm.ppc.altivec.vcmpequb

  res_vbc = vec_cmpeq(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: @llvm.ppc.altivec.vcmpequb

  res_vbs = vec_cmpeq(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: @llvm.ppc.altivec.vcmpequh

  res_vbs = vec_cmpeq(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: @llvm.ppc.altivec.vcmpequh

  res_vbs = vec_cmpeq(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: @llvm.ppc.altivec.vcmpequh

  res_vbs = vec_cmpeq(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: @llvm.ppc.altivec.vcmpequh

  res_vbi = vec_cmpeq(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: @llvm.ppc.altivec.vcmpequw

  res_vbi = vec_cmpeq(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: @llvm.ppc.altivec.vcmpequw

  res_vbi = vec_cmpeq(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: @llvm.ppc.altivec.vcmpequw

  res_vbi = vec_cmpeq(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: @llvm.ppc.altivec.vcmpequw

  res_vbi = vec_cmpeq(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp

  /* vec_cmpne */
  res_vbc = vec_cmpne(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: xor

  res_vbc = vec_cmpne(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: xor

  res_vbc = vec_cmpne(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: xor

  res_vbc = vec_cmpne(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequb
// CHECK-LE: xor

  res_vbs = vec_cmpne(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: xor

  res_vbs = vec_cmpne(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: xor

  res_vbs = vec_cmpne(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: xor

  res_vbs = vec_cmpne(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequh
// CHECK-LE: xor

  res_vbi = vec_cmpne(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: xor

  res_vbi = vec_cmpne(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: xor

  res_vbi = vec_cmpne(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: xor

  res_vbi = vec_cmpne(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpequw
// CHECK-LE: xor

  res_vbi = vec_cmpne(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp
// CHECK: xor
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp
// CHECK-LE: xor

  /* vec_cmpge */
  res_vbc = vec_cmpge(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb

  res_vbc = vec_cmpge(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub

  res_vbs = vec_cmpge(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh

  res_vbs = vec_cmpge(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh

  res_vbi = vec_cmpge(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw

  res_vbi = vec_cmpge(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw

  res_vbi = vec_cmpge(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp

  res_vbi = vec_vcmpgefp(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp
}

// CHECK-LABEL: define{{.*}} void @test5
void test5() {

  /* vec_cmpgt */
  res_vbc = vec_cmpgt(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb

  res_vbc = vec_cmpgt(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub

  res_vbs = vec_cmpgt(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh

  res_vbs = vec_cmpgt(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh

  res_vbi = vec_cmpgt(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw

  res_vbi = vec_cmpgt(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw

  res_vbi = vec_cmpgt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp

  res_vbc = vec_vcmpgtsb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb

  res_vbc = vec_vcmpgtub(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub

  res_vbs = vec_vcmpgtsh(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh

  res_vbs = vec_vcmpgtuh(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh

  res_vbi = vec_vcmpgtsw(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw

  res_vbi = vec_vcmpgtuw(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw

  res_vbi = vec_vcmpgtfp(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp

  /* vec_cmple */
  res_vbc = vec_cmple(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb

  res_vbc = vec_cmple(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub

  res_vbs = vec_cmple(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh

  res_vbs = vec_cmple(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh

  res_vbi = vec_cmple(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw

  res_vbi = vec_cmple(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw

  res_vbi = vec_cmple(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp
}

// CHECK-LABEL: define{{.*}} void @test6
void test6() {
  /* vec_cmplt */
  res_vbc = vec_cmplt(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb

  res_vbc = vec_cmplt(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub

  res_vbs = vec_cmplt(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh

  res_vbs = vec_cmplt(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh

  res_vbi = vec_cmplt(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw

  res_vbi = vec_cmplt(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw

  res_vbi = vec_cmplt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp

  /* vec_ctf */
  res_vf  = vec_ctf(vi, 0);
// CHECK: @llvm.ppc.altivec.vcfsx
// CHECK-LE: @llvm.ppc.altivec.vcfsx

  res_vf  = vec_ctf(vui, 0);
// CHECK: @llvm.ppc.altivec.vcfux
// CHECK-LE: @llvm.ppc.altivec.vcfux

  res_vf  = vec_vcfsx(vi, 0);
// CHECK: @llvm.ppc.altivec.vcfsx
// CHECK-LE: @llvm.ppc.altivec.vcfsx

  res_vf  = vec_vcfux(vui, 0);
// CHECK: @llvm.ppc.altivec.vcfux
// CHECK-LE: @llvm.ppc.altivec.vcfux

  /* vec_cts */
  res_vi = vec_cts(vf, 0);
// CHECK: @llvm.ppc.altivec.vctsxs
// CHECK-LE: @llvm.ppc.altivec.vctsxs

  res_vi = vec_vctsxs(vf, 0);
// CHECK: @llvm.ppc.altivec.vctsxs
// CHECK-LE: @llvm.ppc.altivec.vctsxs

  /* vec_ctu */
  res_vui = vec_ctu(vf, 0);
// CHECK: @llvm.ppc.altivec.vctuxs
// CHECK-LE: @llvm.ppc.altivec.vctuxs

  res_vui = vec_vctuxs(vf, 0);
// CHECK: @llvm.ppc.altivec.vctuxs
// CHECK-LE: @llvm.ppc.altivec.vctuxs

  res_vi = vec_signed(vf);
// CHECK: fptosi <4 x float>
// CHECK-LE: fptosi <4 x float>

  res_vui = vec_unsigned(vf);
// CHECK: fptoui <4 x float>
// CHECK-LE: fptoui <4 x float>

  res_vf = vec_float(vi);
// CHECK: sitofp <4 x i32>
// CHECK-LE: sitofp <4 x i32>

  res_vf = vec_float(vui);
// CHECK: uitofp <4 x i32>
// CHECK-LE: uitofp <4 x i32>

  /* vec_div */
  res_vsc = vec_div(vsc, vsc);
// CHECK: sdiv <16 x i8>
// CHECK-LE: sdiv <16 x i8>

  res_vuc = vec_div(vuc, vuc);
// CHECK: udiv <16 x i8>
// CHECK-LE: udiv <16 x i8>

  res_vs = vec_div(vs, vs);
// CHECK: sdiv <8 x i16>
// CHECK-LE: sdiv <8 x i16>

  res_vus = vec_div(vus, vus);
// CHECK: udiv <8 x i16>
// CHECK-LE: udiv <8 x i16>

  res_vi = vec_div(vi, vi);
// CHECK: sdiv <4 x i32>
// CHECK-LE: sdiv <4 x i32>

  res_vui = vec_div(vui, vui);
// CHECK: udiv <4 x i32>
// CHECK-LE: udiv <4 x i32>

  /* vec_dss */
  vec_dss(0);
// CHECK: @llvm.ppc.altivec.dss
// CHECK-LE: @llvm.ppc.altivec.dss

  /* vec_dssall */
  vec_dssall();
// CHECK: @llvm.ppc.altivec.dssall
// CHECK-LE: @llvm.ppc.altivec.dssall

  /* vec_dst */
  vec_dst(&vsc, 0, 0);
// CHECK: @llvm.ppc.altivec.dst
// CHECK-LE: @llvm.ppc.altivec.dst

  /* vec_dstst */
  vec_dstst(&vs, 0, 0);
// CHECK: @llvm.ppc.altivec.dstst
// CHECK-LE: @llvm.ppc.altivec.dstst

  /* vec_dststt */
  vec_dststt(&param_i, 0, 0);
// CHECK: @llvm.ppc.altivec.dststt
// CHECK-LE: @llvm.ppc.altivec.dststt

  /* vec_dstt */
  vec_dstt(&vf, 0, 0);
// CHECK: @llvm.ppc.altivec.dstt
// CHECK-LE: @llvm.ppc.altivec.dstt

  /* vec_expte */
  res_vf = vec_expte(vf);
// CHECK: @llvm.ppc.altivec.vexptefp
// CHECK-LE: @llvm.ppc.altivec.vexptefp

  res_vf = vec_vexptefp(vf);
// CHECK: @llvm.ppc.altivec.vexptefp
// CHECK-LE: @llvm.ppc.altivec.vexptefp

  /* vec_floor */
  res_vf = vec_floor(vf);
// CHECK: @llvm.ppc.altivec.vrfim
// CHECK-LE: @llvm.ppc.altivec.vrfim

  res_vf = vec_vrfim(vf);
// CHECK: @llvm.ppc.altivec.vrfim
// CHECK-LE: @llvm.ppc.altivec.vrfim

  /* vec_ld */
  res_vsc = vec_ld(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vsc = vec_ld(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vuc = vec_ld(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vuc = vec_ld(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vbc = vec_ld(0, &vbc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vs  = vec_ld(0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vs  = vec_ld(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vus = vec_ld(0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vus = vec_ld(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vbs = vec_ld(0, &vbs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vp  = vec_ld(0, &vp);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vi  = vec_ld(0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vi  = vec_ld(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vui = vec_ld(0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vui = vec_ld(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vbi = vec_ld(0, &vbi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vf  = vec_ld(0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vf  = vec_ld(0, param_f_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vsc = vec_lvx(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vsc = vec_lvx(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vuc = vec_lvx(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vuc = vec_lvx(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vbc = vec_lvx(0, &vbc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vs  = vec_lvx(0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vs  = vec_lvx(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vus = vec_lvx(0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vus = vec_lvx(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vbs = vec_lvx(0, &vbs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vp  = vec_lvx(0, &vp);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vi  = vec_lvx(0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vi  = vec_lvx(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vui = vec_lvx(0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vui = vec_lvx(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vbi = vec_lvx(0, &vbi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vf  = vec_lvx(0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  res_vf  = vec_lvx(0, param_f_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvx

  /* vec_lde */
  res_vsc = vec_lde(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvebx
// CHECK-LE: @llvm.ppc.altivec.lvebx

  res_vuc = vec_lde(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvebx
// CHECK-LE: @llvm.ppc.altivec.lvebx

  res_vs  = vec_lde(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvehx
// CHECK-LE: @llvm.ppc.altivec.lvehx

  res_vus = vec_lde(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvehx
// CHECK-LE: @llvm.ppc.altivec.lvehx

  res_vi  = vec_lde(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvewx
// CHECK-LE: @llvm.ppc.altivec.lvewx

  res_vui = vec_lde(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvewx
// CHECK-LE: @llvm.ppc.altivec.lvewx

  res_vf  = vec_lde(0, param_f_ld);
// CHECK: @llvm.ppc.altivec.lvewx
// CHECK-LE: @llvm.ppc.altivec.lvewx

  res_vsc = vec_lvebx(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvebx
// CHECK-LE: @llvm.ppc.altivec.lvebx

  res_vuc = vec_lvebx(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvebx
// CHECK-LE: @llvm.ppc.altivec.lvebx

  res_vs  = vec_lvehx(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvehx
// CHECK-LE: @llvm.ppc.altivec.lvehx

  res_vus = vec_lvehx(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvehx
// CHECK-LE: @llvm.ppc.altivec.lvehx

  res_vi  = vec_lvewx(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvewx
// CHECK-LE: @llvm.ppc.altivec.lvewx

  res_vui = vec_lvewx(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvewx
// CHECK-LE: @llvm.ppc.altivec.lvewx

  res_vf  = vec_lvewx(0, param_f_ld);
// CHECK: @llvm.ppc.altivec.lvewx
// CHECK-LE: @llvm.ppc.altivec.lvewx

  /* vec_ldl */
  res_vsc = vec_ldl(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vsc = vec_ldl(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vuc = vec_ldl(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vuc = vec_ldl(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vbc = vec_ldl(0, &vbc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vs  = vec_ldl(0, &vs);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vs  = vec_ldl(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vus = vec_ldl(0, &vus);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vus = vec_ldl(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vbs = vec_ldl(0, &vbs);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vp  = vec_ldl(0, &vp);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vi  = vec_ldl(0, &vi);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vi  = vec_ldl(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vui = vec_ldl(0, &vui);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vui = vec_ldl(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vbi = vec_ldl(0, &vbi);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vf  = vec_ldl(0, &vf);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vf  = vec_ldl(0, param_f_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vsc = vec_lvxl(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vsc = vec_lvxl(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vuc = vec_lvxl(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vbc = vec_lvxl(0, &vbc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vuc = vec_lvxl(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vs  = vec_lvxl(0, &vs);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vs  = vec_lvxl(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vus = vec_lvxl(0, &vus);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vus = vec_lvxl(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vbs = vec_lvxl(0, &vbs);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vp  = vec_lvxl(0, &vp);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vi  = vec_lvxl(0, &vi);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vi  = vec_lvxl(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vui = vec_lvxl(0, &vui);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vui = vec_lvxl(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vbi = vec_lvxl(0, &vbi);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vf  = vec_lvxl(0, &vf);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  res_vf  = vec_lvxl(0, param_f_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvxl

  /* vec_loge */
  res_vf = vec_loge(vf);
// CHECK: @llvm.ppc.altivec.vlogefp
// CHECK-LE: @llvm.ppc.altivec.vlogefp

  res_vf = vec_vlogefp(vf);
// CHECK: @llvm.ppc.altivec.vlogefp
// CHECK-LE: @llvm.ppc.altivec.vlogefp

  /* vec_lvsl */
  res_vuc = vec_lvsl(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.lvsl

  /* vec_lvsr */
  res_vuc = vec_lvsr(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.lvsr

  /* vec_madd */
  res_vf =vec_madd(vf, vf, vf);
// CHECK: @llvm.ppc.altivec.vmaddfp
// CHECK-LE: @llvm.ppc.altivec.vmaddfp

  res_vf = vec_vmaddfp(vf, vf, vf);
// CHECK: @llvm.ppc.altivec.vmaddfp
// CHECK-LE: @llvm.ppc.altivec.vmaddfp

  /* vec_madds */
  res_vs = vec_madds(vs, vs, vs);
// CHECK: @llvm.ppc.altivec.vmhaddshs
// CHECK-LE: @llvm.ppc.altivec.vmhaddshs

  res_vs = vec_vmhaddshs(vs, vs, vs);
// CHECK: @llvm.ppc.altivec.vmhaddshs
// CHECK-LE: @llvm.ppc.altivec.vmhaddshs

  /* vec_max */
  res_vsc = vec_max(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  res_vsc = vec_max(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  res_vsc = vec_max(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  res_vuc = vec_max(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vmaxub
// CHECK-LE: @llvm.ppc.altivec.vmaxub

  res_vuc = vec_max(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vmaxub
// CHECK-LE: @llvm.ppc.altivec.vmaxub

  res_vuc = vec_max(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vmaxub
// CHECK-LE: @llvm.ppc.altivec.vmaxub

  res_vs  = vec_max(vs, vs);
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  res_vs  = vec_max(vbs, vs);
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  res_vs  = vec_max(vs, vbs);
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  res_vus = vec_max(vus, vus);
// CHECK: @llvm.ppc.altivec.vmaxuh
// CHECK-LE: @llvm.ppc.altivec.vmaxuh

  res_vus = vec_max(vbs, vus);
// CHECK: @llvm.ppc.altivec.vmaxuh
// CHECK-LE: @llvm.ppc.altivec.vmaxuh

  res_vus = vec_max(vus, vbs);
// CHECK: @llvm.ppc.altivec.vmaxuh
// CHECK-LE: @llvm.ppc.altivec.vmaxuh

  res_vi  = vec_max(vi, vi);
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  res_vi  = vec_max(vbi, vi);
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  res_vi  = vec_max(vi, vbi);
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  res_vui = vec_max(vui, vui);
// CHECK: @llvm.ppc.altivec.vmaxuw
// CHECK-LE: @llvm.ppc.altivec.vmaxuw

  res_vui = vec_max(vbi, vui);
// CHECK: @llvm.ppc.altivec.vmaxuw
// CHECK-LE: @llvm.ppc.altivec.vmaxuw

  res_vui = vec_max(vui, vbi);
// CHECK: @llvm.ppc.altivec.vmaxuw
// CHECK-LE: @llvm.ppc.altivec.vmaxuw

  res_vf  = vec_max(vf, vf);
// CHECK: @llvm.ppc.altivec.vmaxfp
// CHECK-LE: @llvm.ppc.altivec.vmaxfp

  res_vsc = vec_vmaxsb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  res_vsc = vec_vmaxsb(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  res_vsc = vec_vmaxsb(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vmaxsb
// CHECK-LE: @llvm.ppc.altivec.vmaxsb

  res_vuc = vec_vmaxub(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vmaxub
// CHECK-LE: @llvm.ppc.altivec.vmaxub

  res_vuc = vec_vmaxub(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vmaxub
// CHECK-LE: @llvm.ppc.altivec.vmaxub

  res_vuc = vec_vmaxub(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vmaxub
// CHECK-LE: @llvm.ppc.altivec.vmaxub

  res_vs  = vec_vmaxsh(vs, vs);
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  res_vs  = vec_vmaxsh(vbs, vs);
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  res_vs  = vec_vmaxsh(vs, vbs);
// CHECK: @llvm.ppc.altivec.vmaxsh
// CHECK-LE: @llvm.ppc.altivec.vmaxsh

  res_vus = vec_vmaxuh(vus, vus);
// CHECK: @llvm.ppc.altivec.vmaxuh
// CHECK-LE: @llvm.ppc.altivec.vmaxuh

  res_vus = vec_vmaxuh(vbs, vus);
// CHECK: @llvm.ppc.altivec.vmaxuh
// CHECK-LE: @llvm.ppc.altivec.vmaxuh

  res_vus = vec_vmaxuh(vus, vbs);
// CHECK: @llvm.ppc.altivec.vmaxuh
// CHECK-LE: @llvm.ppc.altivec.vmaxuh

  res_vi  = vec_vmaxsw(vi, vi);
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  res_vi  = vec_vmaxsw(vbi, vi);
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  res_vi  = vec_vmaxsw(vi, vbi);
// CHECK: @llvm.ppc.altivec.vmaxsw
// CHECK-LE: @llvm.ppc.altivec.vmaxsw

  res_vui = vec_vmaxuw(vui, vui);
// CHECK: @llvm.ppc.altivec.vmaxuw
// CHECK-LE: @llvm.ppc.altivec.vmaxuw

  res_vui = vec_vmaxuw(vbi, vui);
// CHECK: @llvm.ppc.altivec.vmaxuw
// CHECK-LE: @llvm.ppc.altivec.vmaxuw

  res_vui = vec_vmaxuw(vui, vbi);
// CHECK: @llvm.ppc.altivec.vmaxuw
// CHECK-LE: @llvm.ppc.altivec.vmaxuw

  res_vf  = vec_vmaxfp(vf, vf);
// CHECK: @llvm.ppc.altivec.vmaxfp
// CHECK-LE: @llvm.ppc.altivec.vmaxfp

  /* vec_mergeh */
  res_vsc = vec_mergeh(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_mergeh(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_mergeh(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_mergeh(vs, vs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_mergeh(vp, vp);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_mergeh(vus, vus);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_mergeh(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_mergeh(vi, vi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_mergeh(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_mergeh(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_mergeh(vf, vf);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_vmrghb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_vmrghb(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_vmrghb(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_vmrghh(vs, vs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_vmrghh(vp, vp);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_vmrghh(vus, vus);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_vmrghh(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_vmrghw(vi, vi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_vmrghw(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_vmrghw(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_vmrghw(vf, vf);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_mergel */
  res_vsc = vec_mergel(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_mergel(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_mergel(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_mergel(vs, vs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_mergeh(vp, vp);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_mergel(vus, vus);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_mergel(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_mergel(vi, vi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_mergel(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_mergel(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_mergel(vf, vf);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_vmrglb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_vmrglb(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_vmrglb(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_vmrglh(vs, vs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_vmrglh(vp, vp);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_vmrglh(vus, vus);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_vmrglh(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_vmrglw(vi, vi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_vmrglw(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_vmrglw(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_vmrglw(vf, vf);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_mfvscr */
  vus = vec_mfvscr();
// CHECK: @llvm.ppc.altivec.mfvscr
// CHECK-LE: @llvm.ppc.altivec.mfvscr

  /* vec_min */
  res_vsc = vec_min(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vminsb
// CHECK-LE: @llvm.ppc.altivec.vminsb

  res_vsc = vec_min(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vminsb
// CHECK-LE: @llvm.ppc.altivec.vminsb

  res_vsc = vec_min(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vminsb
// CHECK-LE: @llvm.ppc.altivec.vminsb

  res_vuc = vec_min(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vminub
// CHECK-LE: @llvm.ppc.altivec.vminub

  res_vuc = vec_min(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vminub
// CHECK-LE: @llvm.ppc.altivec.vminub

  res_vuc = vec_min(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vminub
// CHECK-LE: @llvm.ppc.altivec.vminub

  res_vs  = vec_min(vs, vs);
// CHECK: @llvm.ppc.altivec.vminsh
// CHECK-LE: @llvm.ppc.altivec.vminsh

  res_vs  = vec_min(vbs, vs);
// CHECK: @llvm.ppc.altivec.vminsh
// CHECK-LE: @llvm.ppc.altivec.vminsh

  res_vs  = vec_min(vs, vbs);
// CHECK: @llvm.ppc.altivec.vminsh
// CHECK-LE: @llvm.ppc.altivec.vminsh

  res_vus = vec_min(vus, vus);
// CHECK: @llvm.ppc.altivec.vminuh
// CHECK-LE: @llvm.ppc.altivec.vminuh

  res_vus = vec_min(vbs, vus);
// CHECK: @llvm.ppc.altivec.vminuh
// CHECK-LE: @llvm.ppc.altivec.vminuh

  res_vus = vec_min(vus, vbs);
// CHECK: @llvm.ppc.altivec.vminuh
// CHECK-LE: @llvm.ppc.altivec.vminuh

  res_vi  = vec_min(vi, vi);
// CHECK: @llvm.ppc.altivec.vminsw
// CHECK-LE: @llvm.ppc.altivec.vminsw

  res_vi  = vec_min(vbi, vi);
// CHECK: @llvm.ppc.altivec.vminsw
// CHECK-LE: @llvm.ppc.altivec.vminsw

  res_vi  = vec_min(vi, vbi);
// CHECK: @llvm.ppc.altivec.vminsw
// CHECK-LE: @llvm.ppc.altivec.vminsw

  res_vui = vec_min(vui, vui);
// CHECK: @llvm.ppc.altivec.vminuw
// CHECK-LE: @llvm.ppc.altivec.vminuw

  res_vui = vec_min(vbi, vui);
// CHECK: @llvm.ppc.altivec.vminuw
// CHECK-LE: @llvm.ppc.altivec.vminuw

  res_vui = vec_min(vui, vbi);
// CHECK: @llvm.ppc.altivec.vminuw
// CHECK-LE: @llvm.ppc.altivec.vminuw

  res_vf  = vec_min(vf, vf);
// CHECK: @llvm.ppc.altivec.vminfp
// CHECK-LE: @llvm.ppc.altivec.vminfp

  res_vsc = vec_vminsb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vminsb
// CHECK-LE: @llvm.ppc.altivec.vminsb

  res_vsc = vec_vminsb(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vminsb
// CHECK-LE: @llvm.ppc.altivec.vminsb

  res_vsc = vec_vminsb(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vminsb
// CHECK-LE: @llvm.ppc.altivec.vminsb

  res_vuc = vec_vminub(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vminub
// CHECK-LE: @llvm.ppc.altivec.vminub

  res_vuc = vec_vminub(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vminub
// CHECK-LE: @llvm.ppc.altivec.vminub

  res_vuc = vec_vminub(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vminub
// CHECK-LE: @llvm.ppc.altivec.vminub

  res_vs  = vec_vminsh(vs, vs);
// CHECK: @llvm.ppc.altivec.vminsh
// CHECK-LE: @llvm.ppc.altivec.vminsh

  res_vs  = vec_vminsh(vbs, vs);
// CHECK: @llvm.ppc.altivec.vminsh
// CHECK-LE: @llvm.ppc.altivec.vminsh

  res_vs  = vec_vminsh(vs, vbs);
// CHECK: @llvm.ppc.altivec.vminsh
// CHECK-LE: @llvm.ppc.altivec.vminsh

  res_vus = vec_vminuh(vus, vus);
// CHECK: @llvm.ppc.altivec.vminuh
// CHECK-LE: @llvm.ppc.altivec.vminuh

  res_vus = vec_vminuh(vbs, vus);
// CHECK: @llvm.ppc.altivec.vminuh
// CHECK-LE: @llvm.ppc.altivec.vminuh

  res_vus = vec_vminuh(vus, vbs);
// CHECK: @llvm.ppc.altivec.vminuh
// CHECK-LE: @llvm.ppc.altivec.vminuh

  res_vi  = vec_vminsw(vi, vi);
// CHECK: @llvm.ppc.altivec.vminsw
// CHECK-LE: @llvm.ppc.altivec.vminsw

  res_vi  = vec_vminsw(vbi, vi);
// CHECK: @llvm.ppc.altivec.vminsw
// CHECK-LE: @llvm.ppc.altivec.vminsw

  res_vi  = vec_vminsw(vi, vbi);
// CHECK: @llvm.ppc.altivec.vminsw
// CHECK-LE: @llvm.ppc.altivec.vminsw

  res_vui = vec_vminuw(vui, vui);
// CHECK: @llvm.ppc.altivec.vminuw
// CHECK-LE: @llvm.ppc.altivec.vminuw

  res_vui = vec_vminuw(vbi, vui);
// CHECK: @llvm.ppc.altivec.vminuw
// CHECK-LE: @llvm.ppc.altivec.vminuw

  res_vui = vec_vminuw(vui, vbi);
// CHECK: @llvm.ppc.altivec.vminuw
// CHECK-LE: @llvm.ppc.altivec.vminuw

  res_vf  = vec_vminfp(vf, vf);
// CHECK: @llvm.ppc.altivec.vminfp
// CHECK-LE: @llvm.ppc.altivec.vminfp

  /* vec_mladd */
  res_vus = vec_mladd(vus, vus, vus);
// CHECK: mul <8 x i16>
// CHECK: add <8 x i16>
// CHECK-LE: mul <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vs = vec_mladd(vus, vs, vs);
// CHECK: mul <8 x i16>
// CHECK: add <8 x i16>
// CHECK-LE: mul <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vs = vec_mladd(vs, vus, vus);
// CHECK: mul <8 x i16>
// CHECK: add <8 x i16>
// CHECK-LE: mul <8 x i16>
// CHECK-LE: add <8 x i16>

  res_vs = vec_mladd(vs, vs, vs);
// CHECK: mul <8 x i16>
// CHECK: add <8 x i16>
// CHECK-LE: mul <8 x i16>
// CHECK-LE: add <8 x i16>

  /* vec_mradds */
  res_vs = vec_mradds(vs, vs, vs);
// CHECK: @llvm.ppc.altivec.vmhraddshs
// CHECK-LE: @llvm.ppc.altivec.vmhraddshs

  res_vs = vec_vmhraddshs(vs, vs, vs);
// CHECK: @llvm.ppc.altivec.vmhraddshs
// CHECK-LE: @llvm.ppc.altivec.vmhraddshs

  /* vec_msum */
  res_vi  = vec_msum(vsc, vuc, vi);
// CHECK: @llvm.ppc.altivec.vmsummbm
// CHECK-LE: @llvm.ppc.altivec.vmsummbm

  res_vui = vec_msum(vuc, vuc, vui);
// CHECK: @llvm.ppc.altivec.vmsumubm
// CHECK-LE: @llvm.ppc.altivec.vmsumubm

  res_vi  = vec_msum(vs, vs, vi);
// CHECK: @llvm.ppc.altivec.vmsumshm
// CHECK-LE: @llvm.ppc.altivec.vmsumshm

  res_vui = vec_msum(vus, vus, vui);
// CHECK: @llvm.ppc.altivec.vmsumuhm
// CHECK-LE: @llvm.ppc.altivec.vmsumuhm

  res_vi  = vec_vmsummbm(vsc, vuc, vi);
// CHECK: @llvm.ppc.altivec.vmsummbm
// CHECK-LE: @llvm.ppc.altivec.vmsummbm

  res_vui = vec_vmsumubm(vuc, vuc, vui);
// CHECK: @llvm.ppc.altivec.vmsumubm
// CHECK-LE: @llvm.ppc.altivec.vmsumubm

  res_vi  = vec_vmsumshm(vs, vs, vi);
// CHECK: @llvm.ppc.altivec.vmsumshm
// CHECK-LE: @llvm.ppc.altivec.vmsumshm

  res_vui = vec_vmsumuhm(vus, vus, vui);
// CHECK: @llvm.ppc.altivec.vmsumuhm
// CHECK-LE: @llvm.ppc.altivec.vmsumuhm

  /* vec_msums */
  res_vi  = vec_msums(vs, vs, vi);
// CHECK: @llvm.ppc.altivec.vmsumshs
// CHECK-LE: @llvm.ppc.altivec.vmsumshs

  res_vui = vec_msums(vus, vus, vui);
// CHECK: @llvm.ppc.altivec.vmsumuhs
// CHECK-LE: @llvm.ppc.altivec.vmsumuhs

  res_vi  = vec_vmsumshs(vs, vs, vi);
// CHECK: @llvm.ppc.altivec.vmsumshs
// CHECK-LE: @llvm.ppc.altivec.vmsumshs

  res_vui = vec_vmsumuhs(vus, vus, vui);
// CHECK: @llvm.ppc.altivec.vmsumuhs
// CHECK-LE: @llvm.ppc.altivec.vmsumuhs

  /* vec_mtvscr */
  vec_mtvscr(vsc);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vuc);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vbc);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vs);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vus);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vbs);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vp);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vi);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vui);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  vec_mtvscr(vbi);
// CHECK: @llvm.ppc.altivec.mtvscr
// CHECK-LE: @llvm.ppc.altivec.mtvscr

  /* vec_mul */
  res_vsc = vec_mul(vsc, vsc);
// CHECK: mul <16 x i8>
// CHECK-LE: mul <16 x i8>

  res_vuc = vec_mul(vuc, vuc);
// CHECK: mul <16 x i8>
// CHECK-LE: mul <16 x i8>

  res_vs = vec_mul(vs, vs);
// CHECK: mul <8 x i16>
// CHECK-LE: mul <8 x i16>

  res_vus = vec_mul(vus, vus);
// CHECK: mul <8 x i16>
// CHECK-LE: mul <8 x i16>

  res_vi = vec_mul(vi, vi);
// CHECK: mul <4 x i32>
// CHECK-LE: mul <4 x i32>

  res_vui = vec_mul(vui, vui);
// CHECK: mul <4 x i32>
// CHECK-LE: mul <4 x i32>

  /* vec_mule */
  res_vs  = vec_mule(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vmulesb
// CHECK-LE: @llvm.ppc.altivec.vmulosb

  res_vus = vec_mule(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vmuleub
// CHECK-LE: @llvm.ppc.altivec.vmuloub

  res_vi  = vec_mule(vs, vs);
// CHECK: @llvm.ppc.altivec.vmulesh
// CHECK-LE: @llvm.ppc.altivec.vmulosh

  res_vui = vec_mule(vus, vus);
// CHECK: @llvm.ppc.altivec.vmuleuh
// CHECK-LE: @llvm.ppc.altivec.vmulouh

  res_vs  = vec_vmulesb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vmulesb
// CHECK-LE: @llvm.ppc.altivec.vmulosb

  res_vus = vec_vmuleub(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vmuleub
// CHECK-LE: @llvm.ppc.altivec.vmuloub

  res_vi  = vec_vmulesh(vs, vs);
// CHECK: @llvm.ppc.altivec.vmulesh
// CHECK-LE: @llvm.ppc.altivec.vmulosh

  res_vui = vec_vmuleuh(vus, vus);
// CHECK: @llvm.ppc.altivec.vmuleuh
// CHECK-LE: @llvm.ppc.altivec.vmulouh

  /* vec_mulo */
  res_vs  = vec_mulo(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vmulosb
// CHECK-LE: @llvm.ppc.altivec.vmulesb

  res_vus = vec_mulo(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vmuloub
// CHECK-LE: @llvm.ppc.altivec.vmuleub

  res_vi  = vec_mulo(vs, vs);
// CHECK: @llvm.ppc.altivec.vmulosh
// CHECK-LE: @llvm.ppc.altivec.vmulesh

  res_vui = vec_mulo(vus, vus);
// CHECK: @llvm.ppc.altivec.vmulouh
// CHECK-LE: @llvm.ppc.altivec.vmuleuh

  res_vs  = vec_vmulosb(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vmulosb
// CHECK-LE: @llvm.ppc.altivec.vmulesb

  res_vus = vec_vmuloub(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vmuloub
// CHECK-LE: @llvm.ppc.altivec.vmuleub

  res_vi  = vec_vmulosh(vs, vs);
// CHECK: @llvm.ppc.altivec.vmulosh
// CHECK-LE: @llvm.ppc.altivec.vmulesh

  res_vui = vec_vmulouh(vus, vus);
// CHECK: @llvm.ppc.altivec.vmulouh
// CHECK-LE: @llvm.ppc.altivec.vmuleuh

  /* vec_nmsub */
  res_vf = vec_nmsub(vf, vf, vf);
// CHECK: @llvm.ppc.altivec.vnmsubfp
// CHECK-LE: @llvm.ppc.altivec.vnmsubfp

  res_vf = vec_vnmsubfp(vf, vf, vf);
// CHECK: @llvm.ppc.altivec.vnmsubfp
// CHECK-LE: @llvm.ppc.altivec.vnmsubfp

  /* vec_nor */
  res_vsc = vec_nor(vsc, vsc);
// CHECK: or <16 x i8>
// CHECK: xor <16 x i8>
// CHECK-LE: or <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_nor(vuc, vuc);
// CHECK: or <16 x i8>
// CHECK: xor <16 x i8>
// CHECK-LE: or <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vbc = vec_nor(vbc, vbc);
// CHECK: or <16 x i8>
// CHECK: xor <16 x i8>
// CHECK-LE: or <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vs  = vec_nor(vs, vs);
// CHECK: or <8 x i16>
// CHECK: xor <8 x i16>
// CHECK-LE: or <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_nor(vus, vus);
// CHECK: or <8 x i16>
// CHECK: xor <8 x i16>
// CHECK-LE: or <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vbs = vec_nor(vbs, vbs);
// CHECK: or <8 x i16>
// CHECK: xor <8 x i16>
// CHECK-LE: or <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vi  = vec_nor(vi, vi);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_nor(vui, vui);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vbi = vec_nor(vbi, vbi);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_nor(vf, vf);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vsc = vec_vnor(vsc, vsc);
// CHECK: or <16 x i8>
// CHECK: xor <16 x i8>
// CHECK-LE: or <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_vnor(vuc, vuc);
// CHECK: or <16 x i8>
// CHECK: xor <16 x i8>
// CHECK-LE: or <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vbc = vec_vnor(vbc, vbc);
// CHECK: or <16 x i8>
// CHECK: xor <16 x i8>
// CHECK-LE: or <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vs  = vec_vnor(vs, vs);
// CHECK: or <8 x i16>
// CHECK: xor <8 x i16>
// CHECK-LE: or <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_vnor(vus, vus);
// CHECK: or <8 x i16>
// CHECK: xor <8 x i16>
// CHECK-LE: or <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vbs = vec_vnor(vbs, vbs);
// CHECK: or <8 x i16>
// CHECK: xor <8 x i16>
// CHECK-LE: or <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vi  = vec_vnor(vi, vi);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_vnor(vui, vui);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vbi = vec_vnor(vbi, vbi);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_vnor(vf, vf);
// CHECK: or <4 x i32>
// CHECK: xor <4 x i32>
// CHECK-LE: or <4 x i32>
// CHECK-LE: xor <4 x i32>

  /* vec_or */
  res_vsc = vec_or(vsc, vsc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vsc = vec_or(vbc, vsc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vsc = vec_or(vsc, vbc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_or(vuc, vuc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_or(vbc, vuc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_or(vuc, vbc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vbc = vec_or(vbc, vbc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vs  = vec_or(vs, vs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vs  = vec_or(vbs, vs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vs  = vec_or(vs, vbs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_or(vus, vus);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_or(vbs, vus);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_or(vus, vbs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vbs = vec_or(vbs, vbs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vi  = vec_or(vi, vi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vi  = vec_or(vbi, vi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vi  = vec_or(vi, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_or(vui, vui);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_or(vbi, vui);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_or(vui, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vbi = vec_or(vbi, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_or(vf, vf);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_or(vbi, vf);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_or(vf, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vsc = vec_vor(vsc, vsc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vsc = vec_vor(vbc, vsc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vsc = vec_vor(vsc, vbc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_vor(vuc, vuc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_vor(vbc, vuc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_vor(vuc, vbc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vbc = vec_vor(vbc, vbc);
// CHECK: or <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vs  = vec_vor(vs, vs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vs  = vec_vor(vbs, vs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vs  = vec_vor(vs, vbs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_vor(vus, vus);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_vor(vbs, vus);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_vor(vus, vbs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vbs = vec_vor(vbs, vbs);
// CHECK: or <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vi  = vec_vor(vi, vi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vi  = vec_vor(vbi, vi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vi  = vec_vor(vi, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_vor(vui, vui);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_vor(vbi, vui);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_vor(vui, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vbi = vec_vor(vbi, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_vor(vf, vf);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_vor(vbi, vf);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_vor(vf, vbi);
// CHECK: or <4 x i32>
// CHECK-LE: or <4 x i32>

  /* vec_pack */
  res_vsc = vec_pack(vs, vs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_pack(vus, vus);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_pack(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_pack(vi, vi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_pack(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_pack(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_vpkuhum(vs, vs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_vpkuhum(vus, vus);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_vpkuhum(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_vpkuwum(vi, vi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_vpkuwum(vui, vui);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_vpkuwum(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_packpx */
  res_vp = vec_packpx(vui, vui);
// CHECK: @llvm.ppc.altivec.vpkpx
// CHECK-LE: @llvm.ppc.altivec.vpkpx

  res_vp = vec_vpkpx(vui, vui);
// CHECK: @llvm.ppc.altivec.vpkpx
// CHECK-LE: @llvm.ppc.altivec.vpkpx

  /* vec_packs */
  res_vsc = vec_packs(vs, vs);
// CHECK: @llvm.ppc.altivec.vpkshss
// CHECK-LE: @llvm.ppc.altivec.vpkshss

  res_vuc = vec_packs(vus, vus);
// CHECK: @llvm.ppc.altivec.vpkuhus
// CHECK-LE: @llvm.ppc.altivec.vpkuhus

  res_vs  = vec_packs(vi, vi);
// CHECK: @llvm.ppc.altivec.vpkswss
// CHECK-LE: @llvm.ppc.altivec.vpkswss

  res_vus = vec_packs(vui, vui);
// CHECK: @llvm.ppc.altivec.vpkuwus
// CHECK-LE: @llvm.ppc.altivec.vpkuwus

  res_vsc = vec_vpkshss(vs, vs);
// CHECK: @llvm.ppc.altivec.vpkshss
// CHECK-LE: @llvm.ppc.altivec.vpkshss

  res_vuc = vec_vpkuhus(vus, vus);
// CHECK: @llvm.ppc.altivec.vpkuhus
// CHECK-LE: @llvm.ppc.altivec.vpkuhus

  res_vs  = vec_vpkswss(vi, vi);
// CHECK: @llvm.ppc.altivec.vpkswss
// CHECK-LE: @llvm.ppc.altivec.vpkswss

  res_vus = vec_vpkuwus(vui, vui);
// CHECK: @llvm.ppc.altivec.vpkuwus
// CHECK-LE: @llvm.ppc.altivec.vpkuwus

  /* vec_packsu */
  res_vuc = vec_packsu(vs, vs);
// CHECK: @llvm.ppc.altivec.vpkshus
// CHECK-LE: @llvm.ppc.altivec.vpkshus

  res_vuc = vec_packsu(vus, vus);
// CHECK: @llvm.ppc.altivec.vpkuhus
// CHECK-LE: @llvm.ppc.altivec.vpkuhus

  res_vus = vec_packsu(vi, vi);
// CHECK: @llvm.ppc.altivec.vpkswus
// CHECK-LE: @llvm.ppc.altivec.vpkswus

  res_vus = vec_packsu(vui, vui);
// CHECK: @llvm.ppc.altivec.vpkuwus
// CHECK-LE: @llvm.ppc.altivec.vpkuwus

  res_vuc = vec_vpkshus(vs, vs);
// CHECK: @llvm.ppc.altivec.vpkshus
// CHECK-LE: @llvm.ppc.altivec.vpkshus

  res_vuc = vec_vpkshus(vus, vus);
// CHECK: @llvm.ppc.altivec.vpkuhus
// CHECK-LE: @llvm.ppc.altivec.vpkuhus

  res_vus = vec_vpkswus(vi, vi);
// CHECK: @llvm.ppc.altivec.vpkswus
// CHECK-LE: @llvm.ppc.altivec.vpkswus

  res_vus = vec_vpkswus(vui, vui);
// CHECK: @llvm.ppc.altivec.vpkuwus
// CHECK-LE: @llvm.ppc.altivec.vpkuwus

  /* vec_perm */
  res_vsc = vec_perm(vsc, vsc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_perm(vuc, vuc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_perm(vbc, vbc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_perm(vs, vs, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_perm(vus, vus, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_perm(vbs, vbs, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_perm(vp, vp, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_perm(vi, vi, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_perm(vui, vui, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_perm(vbi, vbi, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_perm(vf, vf, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_vperm(vsc, vsc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_vperm(vuc, vuc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_vperm(vbc, vbc, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_vperm(vs, vs, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_vperm(vus, vus, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_vperm(vbs, vbs, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_vperm(vp, vp, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_vperm(vi, vi, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_vperm(vui, vui, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_vperm(vbi, vbi, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_vperm(vf, vf, vuc);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_re */
  res_vf = vec_re(vf);
// CHECK: @llvm.ppc.altivec.vrefp
// CHECK-LE: @llvm.ppc.altivec.vrefp

  res_vf = vec_vrefp(vf);
// CHECK: @llvm.ppc.altivec.vrefp
// CHECK-LE: @llvm.ppc.altivec.vrefp

  /* vec_rl */
  res_vsc = vec_rl(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vrlb
// CHECK-LE: @llvm.ppc.altivec.vrlb

  res_vuc = vec_rl(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vrlb
// CHECK-LE: @llvm.ppc.altivec.vrlb

  res_vs  = vec_rl(vs, vus);
// CHECK: @llvm.ppc.altivec.vrlh
// CHECK-LE: @llvm.ppc.altivec.vrlh

  res_vus = vec_rl(vus, vus);
// CHECK: @llvm.ppc.altivec.vrlh
// CHECK-LE: @llvm.ppc.altivec.vrlh

  res_vi  = vec_rl(vi, vui);
// CHECK: @llvm.ppc.altivec.vrlw
// CHECK-LE: @llvm.ppc.altivec.vrlw

  res_vui = vec_rl(vui, vui);
// CHECK: @llvm.ppc.altivec.vrlw
// CHECK-LE: @llvm.ppc.altivec.vrlw

  res_vsc = vec_vrlb(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vrlb
// CHECK-LE: @llvm.ppc.altivec.vrlb

  res_vuc = vec_vrlb(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vrlb
// CHECK-LE: @llvm.ppc.altivec.vrlb

  res_vs  = vec_vrlh(vs, vus);
// CHECK: @llvm.ppc.altivec.vrlh
// CHECK-LE: @llvm.ppc.altivec.vrlh

  res_vus = vec_vrlh(vus, vus);
// CHECK: @llvm.ppc.altivec.vrlh
// CHECK-LE: @llvm.ppc.altivec.vrlh

  res_vi  = vec_vrlw(vi, vui);
// CHECK: @llvm.ppc.altivec.vrlw
// CHECK-LE: @llvm.ppc.altivec.vrlw

  res_vui = vec_vrlw(vui, vui);
// CHECK: @llvm.ppc.altivec.vrlw
// CHECK-LE: @llvm.ppc.altivec.vrlw

  /* vec_round */
  res_vf = vec_round(vf);
// CHECK: @llvm.ppc.altivec.vrfin
// CHECK-LE: @llvm.ppc.altivec.vrfin

  res_vf = vec_vrfin(vf);
// CHECK: @llvm.ppc.altivec.vrfin
// CHECK-LE: @llvm.ppc.altivec.vrfin

  /* vec_rsqrte */
  res_vf = vec_rsqrte(vf);
// CHECK: @llvm.ppc.altivec.vrsqrtefp
// CHECK-LE: @llvm.ppc.altivec.vrsqrtefp

  res_vf = vec_vrsqrtefp(vf);
// CHECK: @llvm.ppc.altivec.vrsqrtefp
// CHECK-LE: @llvm.ppc.altivec.vrsqrtefp

  /* vec_sel */
  res_vsc = vec_sel(vsc, vsc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vsc = vec_sel(vsc, vsc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_sel(vuc, vuc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_sel(vuc, vuc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vbc = vec_sel(vbc, vbc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vbc = vec_sel(vbc, vbc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vs  = vec_sel(vs, vs, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vs  = vec_sel(vs, vs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_sel(vus, vus, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_sel(vus, vus, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vbs = vec_sel(vbs, vbs, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vbs = vec_sel(vbs, vbs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vi  = vec_sel(vi, vi, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vi  = vec_sel(vi, vi, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_sel(vui, vui, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_sel(vui, vui, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vbi = vec_sel(vbi, vbi, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vbi = vec_sel(vbi, vbi, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_sel(vf, vf, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_sel(vf, vf, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vsc = vec_vsel(vsc, vsc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vsc = vec_vsel(vsc, vsc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_vsel(vuc, vuc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vuc = vec_vsel(vuc, vuc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vbc = vec_vsel(vbc, vbc, vuc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vbc = vec_vsel(vbc, vbc, vbc);
// CHECK: xor <16 x i8>
// CHECK: and <16 x i8>
// CHECK: and <16 x i8>
// CHECK: or <16 x i8>
// CHECK-LE: xor <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: and <16 x i8>
// CHECK-LE: or <16 x i8>

  res_vs  = vec_vsel(vs, vs, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vs  = vec_vsel(vs, vs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_vsel(vus, vus, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vus = vec_vsel(vus, vus, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vbs = vec_vsel(vbs, vbs, vus);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vbs = vec_vsel(vbs, vbs, vbs);
// CHECK: xor <8 x i16>
// CHECK: and <8 x i16>
// CHECK: and <8 x i16>
// CHECK: or <8 x i16>
// CHECK-LE: xor <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: and <8 x i16>
// CHECK-LE: or <8 x i16>

  res_vi  = vec_vsel(vi, vi, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vi  = vec_vsel(vi, vi, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_vsel(vui, vui, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vui = vec_vsel(vui, vui, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vbi = vec_vsel(vbi, vbi, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vbi = vec_vsel(vbi, vbi, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_vsel(vf, vf, vui);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  res_vf  = vec_vsel(vf, vf, vbi);
// CHECK: xor <4 x i32>
// CHECK: and <4 x i32>
// CHECK: and <4 x i32>
// CHECK: or <4 x i32>
// CHECK-LE: xor <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: or <4 x i32>

  /* vec_sl */
  res_vsc = vec_sl(vsc, vuc);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK: shl <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK-LE: shl <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vuc = vec_sl(vuc, vuc);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK: shl <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK-LE: shl <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vs  = vec_sl(vs, vus);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK: shl <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK-LE: shl <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vus = vec_sl(vus, vus);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK: shl <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK-LE: shl <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vi  = vec_sl(vi, vui);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK: shl <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK-LE: shl <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vui = vec_sl(vui, vui);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK: shl <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK-LE: shl <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vsc = vec_vslb(vsc, vuc);
// CHECK: shl <16 x i8>
// CHECK-LE: shl <16 x i8>

  res_vuc = vec_vslb(vuc, vuc);
// CHECK: shl <16 x i8>
// CHECK-LE: shl <16 x i8>

  res_vs  = vec_vslh(vs, vus);
// CHECK: shl <8 x i16>
// CHECK-LE: shl <8 x i16>

  res_vus = vec_vslh(vus, vus);
// CHECK: shl <8 x i16>
// CHECK-LE: shl <8 x i16>

  res_vi  = vec_vslw(vi, vui);
// CHECK: shl <4 x i32>
// CHECK-LE: shl <4 x i32>

  res_vui = vec_vslw(vui, vui);
// CHECK: shl <4 x i32>
// CHECK-LE: shl <4 x i32>

  /* vec_sld */
  res_vsc = vec_sld(vsc, vsc, 0);
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

  res_vuc = vec_sld(vuc, vuc, 0);
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

  res_vs  = vec_sld(vs, vs, 0);
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

  res_vus = vec_sld(vus, vus, 0);
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

  res_vbs = vec_sld(vbs, vbs, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: xor <16 x i8>
// CHECK-LE: [[T1:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: [[T2:%.+]] = bitcast <8 x i16> {{.+}} to <4 x i32>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8>

  res_vp  = vec_sld(vp, vp, 0);
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

  res_vi  = vec_sld(vi, vi, 0);
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

  res_vui = vec_sld(vui, vui, 0);
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

  res_vbi = vec_sld(vbi, vbi, 0);
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 1
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 2
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 3
// CHECK: add nsw i32 {{[0-9a-zA-Z%.]+}}, 15
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{.+}}, <4 x i32> {{.+}}, <16 x i8>
// CHECK-LE: sub nsw i32 16
// CHECK-LE: sub nsw i32 17
// CHECK-LE: sub nsw i32 18
// CHECK-LE: sub nsw i32 31
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{.+}}, <4 x i32> {{.+}}, <16 x i8>

  res_vf  = vec_sld(vf, vf, 0);
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

  /* vec_sldw */
  res_vsc = vec_sldw(vsc, vsc, 0);
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

  res_vuc = vec_sldw(vuc, vuc, 0);
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

  res_vi = vec_sldw(vi, vi, 0);
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

  res_vui = vec_sldw(vui, vui, 0);
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

  res_vs = vec_sldw(vs, vs, 0);
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

  res_vus = vec_sldw(vus, vus, 0);
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

  res_vf = vec_sldw(vf, vf, 0);
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

  res_vsc = vec_vsldoi(vsc, vsc, 0);
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

  res_vuc = vec_vsldoi(vuc, vuc, 0);
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

  res_vs  = vec_vsldoi(vs, vs, 0);
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

  res_vus = vec_vsldoi(vus, vus, 0);
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

  res_vp  = vec_vsldoi(vp, vp, 0);
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

  res_vi  = vec_vsldoi(vi, vi, 0);
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

  res_vui = vec_vsldoi(vui, vui, 0);
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

  res_vf  = vec_vsldoi(vf, vf, 0);
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

  /* vec_sll */
  res_vsc = vec_sll(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vsc = vec_sll(vsc, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vsc = vec_sll(vsc, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vuc = vec_sll(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vuc = vec_sll(vuc, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vuc = vec_sll(vuc, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbc = vec_sll(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbc = vec_sll(vbc, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbc = vec_sll(vbc, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vs  = vec_sll(vs, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vs  = vec_sll(vs, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vs  = vec_sll(vs, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vus = vec_sll(vus, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vus = vec_sll(vus, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vus = vec_sll(vus, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbs = vec_sll(vbs, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbs = vec_sll(vbs, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbs = vec_sll(vbs, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vp  = vec_sll(vp, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vp  = vec_sll(vp, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vp  = vec_sll(vp, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vi  = vec_sll(vi, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vi  = vec_sll(vi, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vi  = vec_sll(vi, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vui = vec_sll(vui, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vui = vec_sll(vui, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vui = vec_sll(vui, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbi = vec_sll(vbi, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbi = vec_sll(vbi, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbi = vec_sll(vbi, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vsc = vec_vsl(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vsc = vec_vsl(vsc, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vsc = vec_vsl(vsc, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vuc = vec_vsl(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vuc = vec_vsl(vuc, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vuc = vec_vsl(vuc, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbc = vec_vsl(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbc = vec_vsl(vbc, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbc = vec_vsl(vbc, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vs  = vec_vsl(vs, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vs  = vec_vsl(vs, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vs  = vec_vsl(vs, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vus = vec_vsl(vus, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vus = vec_vsl(vus, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vus = vec_vsl(vus, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbs = vec_vsl(vbs, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbs = vec_vsl(vbs, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbs = vec_vsl(vbs, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vp  = vec_vsl(vp, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vp  = vec_vsl(vp, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vp  = vec_vsl(vp, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vi  = vec_vsl(vi, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vi  = vec_vsl(vi, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vi  = vec_vsl(vi, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vui = vec_vsl(vui, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vui = vec_vsl(vui, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vui = vec_vsl(vui, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbi = vec_vsl(vbi, vuc);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbi = vec_vsl(vbi, vus);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  res_vbi = vec_vsl(vbi, vui);
// CHECK: @llvm.ppc.altivec.vsl
// CHECK-LE: @llvm.ppc.altivec.vsl

  /* vec_slo */
  res_vsc = vec_slo(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vsc = vec_slo(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vuc = vec_slo(vuc, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vuc = vec_slo(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vs  = vec_slo(vs, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vs  = vec_slo(vs, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vus = vec_slo(vus, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vus = vec_slo(vus, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vp  = vec_slo(vp, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vp  = vec_slo(vp, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vi  = vec_slo(vi, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vi  = vec_slo(vi, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vui = vec_slo(vui, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vui = vec_slo(vui, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vf  = vec_slo(vf, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vf  = vec_slo(vf, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vsc = vec_vslo(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vsc = vec_vslo(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vuc = vec_vslo(vuc, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vuc = vec_vslo(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vs  = vec_vslo(vs, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vs  = vec_vslo(vs, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vus = vec_vslo(vus, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vus = vec_vslo(vus, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vp  = vec_vslo(vp, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vp  = vec_vslo(vp, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vi  = vec_vslo(vi, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vi  = vec_vslo(vi, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vui = vec_vslo(vui, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vui = vec_vslo(vui, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vf  = vec_vslo(vf, vsc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  res_vf  = vec_vslo(vf, vuc);
// CHECK: @llvm.ppc.altivec.vslo
// CHECK-LE: @llvm.ppc.altivec.vslo

  /* vec_splat */
  res_vsc = vec_splat(vsc, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_splat(vuc, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_splat(vbc, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_splat(vs, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_splat(vus, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_splat(vbs, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_splat(vp, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_splat(vi, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_splat(vui, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_splat(vbi, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_splat(vf, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_vspltb(vsc, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_vspltb(vuc, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_vspltb(vbc, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_vsplth(vs, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_vsplth(vus, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_vsplth(vbs, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_vsplth(vp, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_vspltw(vi, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_vspltw(vui, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_vspltw(vbi, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_vspltw(vf, 0);
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_splat_s8 */
  res_vsc = vec_splat_s8(0x09);                 // TODO: add check
  res_vsc = vec_vspltisb(0x09);                 // TODO: add check

  /* vec_splat_s16 */
  res_vs = vec_splat_s16(0x09);                 // TODO: add check
  res_vs = vec_vspltish(0x09);                  // TODO: add check

  /* vec_splat_s32 */
  res_vi = vec_splat_s32(0x09);                 // TODO: add check
  res_vi = vec_vspltisw(0x09);                  // TODO: add check

  /* vec_splat_u8 */
  res_vuc = vec_splat_u8(0x09);                 // TODO: add check

  /* vec_splat_u16 */
  res_vus = vec_splat_u16(0x09);                // TODO: add check

  /* vec_splat_u32 */
  res_vui = vec_splat_u32(0x09);                // TODO: add check

  /* vec_sr */
  res_vsc = vec_sr(vsc, vuc);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK-LE: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vuc = vec_sr(vuc, vuc);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK-LE: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vs  = vec_sr(vs, vus);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK-LE: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vus = vec_sr(vus, vus);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK-LE: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vi  = vec_sr(vi, vui);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK-LE: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vui = vec_sr(vui, vui);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK-LE: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vsc = vec_vsrb(vsc, vuc);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK-LE: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vuc = vec_vsrb(vuc, vuc);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <16 x i8> {{[0-9a-zA-Z%.]+}}, splat (i8 8)
// CHECK-LE: lshr <16 x i8> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vs  = vec_vsrh(vs, vus);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK-LE: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vus = vec_vsrh(vus, vus);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <8 x i16> {{[0-9a-zA-Z%.]+}}, splat (i16 16)
// CHECK-LE: lshr <8 x i16> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vi  = vec_vsrw(vi, vui);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK-LE: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  res_vui = vec_vsrw(vui, vui);
// CHECK: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]
// CHECK-LE: [[UREM:[0-9a-zA-Z%.]+]] = urem <4 x i32> {{[0-9a-zA-Z%.]+}}, splat (i32 32)
// CHECK-LE: lshr <4 x i32> {{[0-9a-zA-Z%.]+}}, [[UREM]]

  /* vec_sra */
  res_vsc = vec_sra(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsrab
// CHECK-LE: @llvm.ppc.altivec.vsrab

  res_vuc = vec_sra(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsrab
// CHECK-LE: @llvm.ppc.altivec.vsrab

  res_vs  = vec_sra(vs, vus);
// CHECK: @llvm.ppc.altivec.vsrah
// CHECK-LE: @llvm.ppc.altivec.vsrah

  res_vus = vec_sra(vus, vus);
// CHECK: @llvm.ppc.altivec.vsrah
// CHECK-LE: @llvm.ppc.altivec.vsrah

  res_vi  = vec_sra(vi, vui);
// CHECK: @llvm.ppc.altivec.vsraw
// CHECK-LE: @llvm.ppc.altivec.vsraw

  res_vui = vec_sra(vui, vui);
// CHECK: @llvm.ppc.altivec.vsraw
// CHECK-LE: @llvm.ppc.altivec.vsraw

  res_vsc = vec_vsrab(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsrab
// CHECK-LE: @llvm.ppc.altivec.vsrab

  res_vuc = vec_vsrab(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsrab
// CHECK-LE: @llvm.ppc.altivec.vsrab

  res_vs  = vec_vsrah(vs, vus);
// CHECK: @llvm.ppc.altivec.vsrah
// CHECK-LE: @llvm.ppc.altivec.vsrah

  res_vus = vec_vsrah(vus, vus);
// CHECK: @llvm.ppc.altivec.vsrah
// CHECK-LE: @llvm.ppc.altivec.vsrah

  res_vi  = vec_vsraw(vi, vui);
// CHECK: @llvm.ppc.altivec.vsraw
// CHECK-LE: @llvm.ppc.altivec.vsraw

  res_vui = vec_vsraw(vui, vui);
// CHECK: @llvm.ppc.altivec.vsraw
// CHECK-LE: @llvm.ppc.altivec.vsraw

  /* vec_srl */
  res_vsc = vec_srl(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vsc = vec_srl(vsc, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vsc = vec_srl(vsc, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vuc = vec_srl(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vuc = vec_srl(vuc, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vuc = vec_srl(vuc, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbc = vec_srl(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbc = vec_srl(vbc, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbc = vec_srl(vbc, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vs  = vec_srl(vs, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vs  = vec_srl(vs, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vs  = vec_srl(vs, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vus = vec_srl(vus, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vus = vec_srl(vus, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vus = vec_srl(vus, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbs = vec_srl(vbs, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbs = vec_srl(vbs, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbs = vec_srl(vbs, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vp  = vec_srl(vp, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vp  = vec_srl(vp, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vp  = vec_srl(vp, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vi  = vec_srl(vi, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vi  = vec_srl(vi, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vi  = vec_srl(vi, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vui = vec_srl(vui, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vui = vec_srl(vui, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vui = vec_srl(vui, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbi = vec_srl(vbi, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbi = vec_srl(vbi, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbi = vec_srl(vbi, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vsc = vec_vsr(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vsc = vec_vsr(vsc, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vsc = vec_vsr(vsc, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vuc = vec_vsr(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vuc = vec_vsr(vuc, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vuc = vec_vsr(vuc, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbc = vec_vsr(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbc = vec_vsr(vbc, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbc = vec_vsr(vbc, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vs  = vec_vsr(vs, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vs  = vec_vsr(vs, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vs  = vec_vsr(vs, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vus = vec_vsr(vus, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vus = vec_vsr(vus, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vus = vec_vsr(vus, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbs = vec_vsr(vbs, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbs = vec_vsr(vbs, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbs = vec_vsr(vbs, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vp  = vec_vsr(vp, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vp  = vec_vsr(vp, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vp  = vec_vsr(vp, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vi  = vec_vsr(vi, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vi  = vec_vsr(vi, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vi  = vec_vsr(vi, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vui = vec_vsr(vui, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vui = vec_vsr(vui, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vui = vec_vsr(vui, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbi = vec_vsr(vbi, vuc);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbi = vec_vsr(vbi, vus);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  res_vbi = vec_vsr(vbi, vui);
// CHECK: @llvm.ppc.altivec.vsr
// CHECK-LE: @llvm.ppc.altivec.vsr

  /* vec_sro */
  res_vsc = vec_sro(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vsc = vec_sro(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vuc = vec_sro(vuc, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vuc = vec_sro(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vs  = vec_sro(vs, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vs  = vec_sro(vs, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vus = vec_sro(vus, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vus = vec_sro(vus, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vp  = vec_sro(vp, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vp  = vec_sro(vp, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vi  = vec_sro(vi, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vi  = vec_sro(vi, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vui = vec_sro(vui, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vui = vec_sro(vui, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vf  = vec_sro(vf, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vf  = vec_sro(vf, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vsc = vec_vsro(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vsc = vec_vsro(vsc, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vuc = vec_vsro(vuc, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vuc = vec_vsro(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vs  = vec_vsro(vs, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vs  = vec_vsro(vs, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vus = vec_vsro(vus, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vus = vec_vsro(vus, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vp  = vec_vsro(vp, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vp  = vec_vsro(vp, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vi  = vec_vsro(vi, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vi  = vec_vsro(vi, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vui = vec_vsro(vui, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vui = vec_vsro(vui, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vf  = vec_vsro(vf, vsc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  res_vf  = vec_vsro(vf, vuc);
// CHECK: @llvm.ppc.altivec.vsro
// CHECK-LE: @llvm.ppc.altivec.vsro

  /* vec_st */
  vec_st(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbc, 0, &vbc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbs, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbs, 0, &vbs);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vp, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vp, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vp, 0, &vp);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbi, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vbi, 0, &vbi);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_st(vf, 0, &param_f);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbc, 0, &vbc);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbs, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbs, 0, &vbs);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vp, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vp, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vp, 0, &vp);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbi, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vbi, 0, &vbi);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvx(vf, 0, &param_f);
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.stvx

  /* vec_ste */
  vec_ste(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_ste(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_ste(vbc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_ste(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_ste(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_ste(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_ste(vbs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_ste(vbs, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_ste(vp, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_ste(vp, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_ste(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_ste(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_ste(vbi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_ste(vbi, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_ste(vf, 0, &param_f);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_stvebx(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_stvebx(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_stvebx(vbc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_stvebx(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvebx
// CHECK-LE: @llvm.ppc.altivec.stvebx

  vec_stvehx(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_stvehx(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_stvehx(vbs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_stvehx(vbs, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_stvehx(vp, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_stvehx(vp, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvehx
// CHECK-LE: @llvm.ppc.altivec.stvehx

  vec_stvewx(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_stvewx(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_stvewx(vbi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_stvewx(vbi, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  vec_stvewx(vf, 0, &param_f);
// CHECK: @llvm.ppc.altivec.stvewx
// CHECK-LE: @llvm.ppc.altivec.stvewx

  /* vec_stl */
  vec_stl(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbc, 0, &vbc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbs, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbs, 0, &vbs);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vp, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vp, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vp, 0, &vp);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbi, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vbi, 0, &vbi);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stl(vf, 0, &param_f);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbc, 0, &vbc);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbs, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbs, 0, &vbs);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vp, 0, &param_s);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vp, 0, &param_us);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vp, 0, &vp);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbi, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vbi, 0, &vbi);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvxl(vf, 0, &param_f);
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.stvxl

  /* vec_sub */
  res_vsc = vec_sub(vsc, vsc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vsc = vec_sub(vbc, vsc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vsc = vec_sub(vsc, vbc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vuc = vec_sub(vuc, vuc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vuc = vec_sub(vbc, vuc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vuc = vec_sub(vuc, vbc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vs  = vec_sub(vs, vs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vs  = vec_sub(vbs, vs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vs  = vec_sub(vs, vbs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vus = vec_sub(vus, vus);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vus = vec_sub(vbs, vus);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vus = vec_sub(vus, vbs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vi  = vec_sub(vi, vi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vi  = vec_sub(vbi, vi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vi  = vec_sub(vi, vbi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vui = vec_sub(vui, vui);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vui = vec_sub(vbi, vui);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vui = vec_sub(vui, vbi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vf  = vec_sub(vf, vf);
// CHECK: fsub <4 x float>
// CHECK-LE: fsub <4 x float>

  

  res_vsc = vec_vsububm(vsc, vsc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vsc = vec_vsububm(vbc, vsc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vsc = vec_vsububm(vsc, vbc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vuc = vec_vsububm(vuc, vuc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vuc = vec_vsububm(vbc, vuc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vuc = vec_vsububm(vuc, vbc);
// CHECK: sub <16 x i8>
// CHECK-LE: sub <16 x i8>

  res_vs  = vec_vsubuhm(vs, vs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vs  = vec_vsubuhm(vbs, vs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vs  = vec_vsubuhm(vs, vbs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vus = vec_vsubuhm(vus, vus);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vus = vec_vsubuhm(vbs, vus);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vus = vec_vsubuhm(vus, vbs);
// CHECK: sub <8 x i16>
// CHECK-LE: sub <8 x i16>

  res_vi  = vec_vsubuwm(vi, vi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vi  = vec_vsubuwm(vbi, vi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vi  = vec_vsubuwm(vi, vbi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vui = vec_vsubuwm(vui, vui);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vui = vec_vsubuwm(vbi, vui);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vui = vec_vsubuwm(vui, vbi);
// CHECK: sub <4 x i32>
// CHECK-LE: sub <4 x i32>

  res_vf  = vec_vsubfp(vf, vf);
// CHECK: fsub <4 x float>
// CHECK-LE: fsub <4 x float>

  /* vec_subc */
  res_vui = vec_subc(vui, vui);
// CHECK: @llvm.ppc.altivec.vsubcuw
// CHECK-LE: @llvm.ppc.altivec.vsubcuw

  res_vi = vec_subc(vi, vi);
// CHECK: @llvm.ppc.altivec.vsubcuw
// CHECK-LE: @llvm.ppc.altivec.vsubcuw

  res_vui = vec_vsubcuw(vui, vui);
// CHECK: @llvm.ppc.altivec.vsubcuw
// CHECK-LE: @llvm.ppc.altivec.vsubcuw

  /* vec_subs */
  res_vsc = vec_subs(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vsubsbs
// CHECK-LE: @llvm.ppc.altivec.vsubsbs

  res_vsc = vec_subs(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vsubsbs
// CHECK-LE: @llvm.ppc.altivec.vsubsbs

  res_vsc = vec_subs(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vsubsbs
// CHECK-LE: @llvm.ppc.altivec.vsubsbs

  res_vuc = vec_subs(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsububs
// CHECK-LE: @llvm.ppc.altivec.vsububs

  res_vuc = vec_subs(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vsububs
// CHECK-LE: @llvm.ppc.altivec.vsububs

  res_vuc = vec_subs(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vsububs
// CHECK-LE: @llvm.ppc.altivec.vsububs

  res_vs  = vec_subs(vs, vs);
// CHECK: @llvm.ppc.altivec.vsubshs
// CHECK-LE: @llvm.ppc.altivec.vsubshs

  res_vs  = vec_subs(vbs, vs);
// CHECK: @llvm.ppc.altivec.vsubshs
// CHECK-LE: @llvm.ppc.altivec.vsubshs

  res_vs  = vec_subs(vs, vbs);
// CHECK: @llvm.ppc.altivec.vsubshs
// CHECK-LE: @llvm.ppc.altivec.vsubshs

  res_vus = vec_subs(vus, vus);
// CHECK: @llvm.ppc.altivec.vsubuhs
// CHECK-LE: @llvm.ppc.altivec.vsubuhs

  res_vus = vec_subs(vbs, vus);
// CHECK: @llvm.ppc.altivec.vsubuhs
// CHECK-LE: @llvm.ppc.altivec.vsubuhs

  res_vus = vec_subs(vus, vbs);
// CHECK: @llvm.ppc.altivec.vsubuhs
// CHECK-LE: @llvm.ppc.altivec.vsubuhs

  res_vi  = vec_subs(vi, vi);
// CHECK: @llvm.ppc.altivec.vsubsws
// CHECK-LE: @llvm.ppc.altivec.vsubsws

  res_vi  = vec_subs(vbi, vi);
// CHECK: @llvm.ppc.altivec.vsubsws
// CHECK-LE: @llvm.ppc.altivec.vsubsws

  res_vi  = vec_subs(vi, vbi);
// CHECK: @llvm.ppc.altivec.vsubsws
// CHECK-LE: @llvm.ppc.altivec.vsubsws

  res_vui = vec_subs(vui, vui);
// CHECK: @llvm.ppc.altivec.vsubuws
// CHECK-LE: @llvm.ppc.altivec.vsubuws

  res_vui = vec_subs(vbi, vui);
// CHECK: @llvm.ppc.altivec.vsubuws
// CHECK-LE: @llvm.ppc.altivec.vsubuws

  res_vui = vec_subs(vui, vbi);
// CHECK: @llvm.ppc.altivec.vsubuws
// CHECK-LE: @llvm.ppc.altivec.vsubuws

  res_vi = vec_sube(vi, vi, vi);
// CHECK: and <4 x i32>
// CHECK: xor <4 x i32> {{%[0-9]+}}, splat (i32 -1)
// CHECK: add <4 x i32>
// CHECK: add <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: xor <4 x i32> {{%[0-9]+}}, splat (i32 -1)
// CHECK-LE: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vui = vec_sube(vui, vui, vui);
// CHECK: and <4 x i32>
// CHECK: xor <4 x i32> {{%[0-9]+}}, splat (i32 -1)
// CHECK: add <4 x i32>
// CHECK: add <4 x i32>
// CHECK-LE: and <4 x i32>
// CHECK-LE: xor <4 x i32> {{%[0-9]+}}, splat (i32 -1)
// CHECK-LE: add <4 x i32>
// CHECK-LE: add <4 x i32>

  res_vsc = vec_vsubsbs(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vsubsbs
// CHECK-LE: @llvm.ppc.altivec.vsubsbs

  res_vsc = vec_vsubsbs(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vsubsbs
// CHECK-LE: @llvm.ppc.altivec.vsubsbs

  res_vsc = vec_vsubsbs(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vsubsbs
// CHECK-LE: @llvm.ppc.altivec.vsubsbs

  res_vuc = vec_vsububs(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vsububs
// CHECK-LE: @llvm.ppc.altivec.vsububs

  res_vuc = vec_vsububs(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vsububs
// CHECK-LE: @llvm.ppc.altivec.vsububs

  res_vuc = vec_vsububs(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vsububs
// CHECK-LE: @llvm.ppc.altivec.vsububs

  res_vs  = vec_vsubshs(vs, vs);
// CHECK: @llvm.ppc.altivec.vsubshs
// CHECK-LE: @llvm.ppc.altivec.vsubshs

  res_vs  = vec_vsubshs(vbs, vs);
// CHECK: @llvm.ppc.altivec.vsubshs
// CHECK-LE: @llvm.ppc.altivec.vsubshs

  res_vs  = vec_vsubshs(vs, vbs);
// CHECK: @llvm.ppc.altivec.vsubshs
// CHECK-LE: @llvm.ppc.altivec.vsubshs

  res_vus = vec_vsubuhs(vus, vus);
// CHECK: @llvm.ppc.altivec.vsubuhs
// CHECK-LE: @llvm.ppc.altivec.vsubuhs

  res_vus = vec_vsubuhs(vbs, vus);
// CHECK: @llvm.ppc.altivec.vsubuhs
// CHECK-LE: @llvm.ppc.altivec.vsubuhs

  res_vus = vec_vsubuhs(vus, vbs);
// CHECK: @llvm.ppc.altivec.vsubuhs
// CHECK-LE: @llvm.ppc.altivec.vsubuhs

  res_vi  = vec_vsubsws(vi, vi);
// CHECK: @llvm.ppc.altivec.vsubsws
// CHECK-LE: @llvm.ppc.altivec.vsubsws

  res_vi  = vec_vsubsws(vbi, vi);
// CHECK: @llvm.ppc.altivec.vsubsws
// CHECK-LE: @llvm.ppc.altivec.vsubsws

  res_vi  = vec_vsubsws(vi, vbi);
// CHECK: @llvm.ppc.altivec.vsubsws
// CHECK-LE: @llvm.ppc.altivec.vsubsws

  res_vui = vec_vsubuws(vui, vui);
// CHECK: @llvm.ppc.altivec.vsubuws
// CHECK-LE: @llvm.ppc.altivec.vsubuws

  res_vui = vec_vsubuws(vbi, vui);
// CHECK: @llvm.ppc.altivec.vsubuws
// CHECK-LE: @llvm.ppc.altivec.vsubuws

  res_vui = vec_vsubuws(vui, vbi);
// CHECK: @llvm.ppc.altivec.vsubuws
// CHECK-LE: @llvm.ppc.altivec.vsubuws

  /* vec_sum4s */
  res_vi  = vec_sum4s(vsc, vi);
// CHECK: @llvm.ppc.altivec.vsum4sbs
// CHECK-LE: @llvm.ppc.altivec.vsum4sbs

  res_vui = vec_sum4s(vuc, vui);
// CHECK: @llvm.ppc.altivec.vsum4ubs
// CHECK-LE: @llvm.ppc.altivec.vsum4ubs

  res_vi  = vec_sum4s(vs, vi);
// CHECK: @llvm.ppc.altivec.vsum4shs
// CHECK-LE: @llvm.ppc.altivec.vsum4shs

  res_vi  = vec_vsum4sbs(vsc, vi);
// CHECK: @llvm.ppc.altivec.vsum4sbs
// CHECK-LE: @llvm.ppc.altivec.vsum4sbs

  res_vui = vec_vsum4ubs(vuc, vui);
// CHECK: @llvm.ppc.altivec.vsum4ubs
// CHECK-LE: @llvm.ppc.altivec.vsum4ubs

  res_vi  = vec_vsum4shs(vs, vi);
// CHECK: @llvm.ppc.altivec.vsum4shs
// CHECK-LE: @llvm.ppc.altivec.vsum4shs

  /* vec_sum2s */
  res_vi = vec_sum2s(vi, vi);
// CHECK: @llvm.ppc.altivec.vsum2sws
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vsum2sws
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi = vec_vsum2sws(vi, vi);
// CHECK: @llvm.ppc.altivec.vsum2sws
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vsum2sws
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_sums */
  res_vi = vec_sums(vi, vi);
// CHECK: @llvm.ppc.altivec.vsumsws
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vsumsws

  res_vi = vec_vsumsws(vi, vi);
// CHECK: @llvm.ppc.altivec.vsumsws
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.vsumsws

  /* vec_trunc */
  res_vf = vec_trunc(vf);
// CHECK: @llvm.ppc.altivec.vrfiz
// CHECK-LE: @llvm.ppc.altivec.vrfiz

  res_vf = vec_vrfiz(vf);
// CHECK: @llvm.ppc.altivec.vrfiz
// CHECK-LE: @llvm.ppc.altivec.vrfiz

  /* vec_unpackh */
  res_vs  = vec_unpackh(vsc);
// CHECK: @llvm.ppc.altivec.vupkhsb
// CHECK-LE: @llvm.ppc.altivec.vupklsb

  res_vbs = vec_unpackh(vbc);
// CHECK: @llvm.ppc.altivec.vupkhsb
// CHECK-LE: @llvm.ppc.altivec.vupklsb

  res_vi  = vec_unpackh(vs);
// CHECK: @llvm.ppc.altivec.vupkhsh
// CHECK-LE: @llvm.ppc.altivec.vupklsh

  res_vbi = vec_unpackh(vbs);
// CHECK: @llvm.ppc.altivec.vupkhsh
// CHECK-LE: @llvm.ppc.altivec.vupklsh

  res_vui = vec_unpackh(vp);
// CHECK: @llvm.ppc.altivec.vupkhpx
// CHECK-LE: @llvm.ppc.altivec.vupklpx

  res_vui = vec_vupkhpx(vp);
// CHECK: @llvm.ppc.altivec.vupkhpx
// CHECK-LE: @llvm.ppc.altivec.vupklpx

  res_vs  = vec_vupkhsb(vsc);
// CHECK: @llvm.ppc.altivec.vupkhsb
// CHECK-LE: @llvm.ppc.altivec.vupklsb

  res_vbs = vec_vupkhsb(vbc);
// CHECK: @llvm.ppc.altivec.vupkhsb
// CHECK-LE: @llvm.ppc.altivec.vupklsb

  res_vi  = vec_vupkhsh(vs);
// CHECK: @llvm.ppc.altivec.vupkhsh
// CHECK-LE: @llvm.ppc.altivec.vupklsh

  res_vbi = vec_vupkhsh(vbs);
// CHECK: @llvm.ppc.altivec.vupkhsh
// CHECK-LE: @llvm.ppc.altivec.vupklsh

  res_vui = vec_vupkhsh(vp);
// CHECK: @llvm.ppc.altivec.vupkhpx
// CHECK-LE: @llvm.ppc.altivec.vupklpx

  /* vec_unpackl */
  res_vs  = vec_unpackl(vsc);
// CHECK: @llvm.ppc.altivec.vupklsb
// CHECK-LE: @llvm.ppc.altivec.vupkhsb

  res_vbs = vec_unpackl(vbc);
// CHECK: @llvm.ppc.altivec.vupklsb
// CHECK-LE: @llvm.ppc.altivec.vupkhsb

  res_vi  = vec_unpackl(vs);
// CHECK: @llvm.ppc.altivec.vupklsh
// CHECK-LE: @llvm.ppc.altivec.vupkhsh

  res_vbi = vec_unpackl(vbs);
// CHECK: @llvm.ppc.altivec.vupklsh
// CHECK-LE: @llvm.ppc.altivec.vupkhsh

  res_vui = vec_unpackl(vp);
// CHECK: @llvm.ppc.altivec.vupklpx
// CHECK-LE: @llvm.ppc.altivec.vupkhpx

  res_vui = vec_vupklpx(vp);
// CHECK: @llvm.ppc.altivec.vupklpx
// CHECK-LE: @llvm.ppc.altivec.vupkhpx

  res_vs  = vec_vupklsb(vsc);
// CHECK: @llvm.ppc.altivec.vupklsb
// CHECK-LE: @llvm.ppc.altivec.vupkhsb

  res_vbs = vec_vupklsb(vbc);
// CHECK: @llvm.ppc.altivec.vupklsb
// CHECK-LE: @llvm.ppc.altivec.vupkhsb

  res_vi  = vec_vupklsh(vs);
// CHECK: @llvm.ppc.altivec.vupklsh
// CHECK-LE: @llvm.ppc.altivec.vupkhsh

  res_vbi = vec_vupklsh(vbs);
// CHECK: @llvm.ppc.altivec.vupklsh
// CHECK-LE: @llvm.ppc.altivec.vupkhsh

  res_vui = vec_vupklsh(vp);
// CHECK: @llvm.ppc.altivec.vupklpx
// CHECK-LE: @llvm.ppc.altivec.vupkhpx

  /* vec_xor */
  res_vsc = vec_xor(vsc, vsc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vsc = vec_xor(vbc, vsc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vsc = vec_xor(vsc, vbc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_xor(vuc, vuc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_xor(vbc, vuc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_xor(vuc, vbc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vbc = vec_xor(vbc, vbc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vs  = vec_xor(vs, vs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vs  = vec_xor(vbs, vs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vs  = vec_xor(vs, vbs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_xor(vus, vus);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_xor(vbs, vus);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_xor(vus, vbs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vbs = vec_xor(vbs, vbs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vi  = vec_xor(vi, vi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vi  = vec_xor(vbi, vi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vi  = vec_xor(vi, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_xor(vui, vui);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_xor(vbi, vui);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_xor(vui, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vbi = vec_xor(vbi, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_xor(vf, vf);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_xor(vbi, vf);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_xor(vf, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vsc = vec_vxor(vsc, vsc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vsc = vec_vxor(vbc, vsc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vsc = vec_vxor(vsc, vbc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_vxor(vuc, vuc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_vxor(vbc, vuc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vuc = vec_vxor(vuc, vbc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vbc = vec_vxor(vbc, vbc);
// CHECK: xor <16 x i8>
// CHECK-LE: xor <16 x i8>

  res_vs  = vec_vxor(vs, vs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vs  = vec_vxor(vbs, vs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vs  = vec_vxor(vs, vbs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_vxor(vus, vus);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_vxor(vbs, vus);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vus = vec_vxor(vus, vbs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vbs = vec_vxor(vbs, vbs);
// CHECK: xor <8 x i16>
// CHECK-LE: xor <8 x i16>

  res_vi  = vec_vxor(vi, vi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vi  = vec_vxor(vbi, vi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vi  = vec_vxor(vi, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_vxor(vui, vui);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_vxor(vbi, vui);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vui = vec_vxor(vui, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vbi = vec_vxor(vbi, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_vxor(vf, vf);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_vxor(vbi, vf);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  res_vf  = vec_vxor(vf, vbi);
// CHECK: xor <4 x i32>
// CHECK-LE: xor <4 x i32>

  /* ------------------------------ extensions -------------------------------------- */

  /* vec_extract */
  res_sc = vec_extract(vsc, param_i);
// CHECK: extractelement <16 x i8>
// CHECK-LE: extractelement <16 x i8>

  res_uc = vec_extract(vuc, param_i);
// CHECK: extractelement <16 x i8>
// CHECK-LE: extractelement <16 x i8>

  res_uc = vec_extract(vbc, param_i);
// CHECK: extractelement <16 x i8>
// CHECK-LE: extractelement <16 x i8>

  res_s  = vec_extract(vs, param_i);
// CHECK: extractelement <8 x i16>
// CHECK-LE: extractelement <8 x i16>

  res_us = vec_extract(vus, param_i);
// CHECK: extractelement <8 x i16>
// CHECK-LE: extractelement <8 x i16>

  res_us = vec_extract(vbs, param_i);
// CHECK: extractelement <8 x i16>
// CHECK-LE: extractelement <8 x i16>

  res_i  = vec_extract(vi, param_i);
// CHECK: extractelement <4 x i32>
// CHECK-LE: extractelement <4 x i32>

  res_ui = vec_extract(vui, param_i);
// CHECK: extractelement <4 x i32>
// CHECK-LE: extractelement <4 x i32>

  res_ui = vec_extract(vbi, param_i);
// CHECK: extractelement <4 x i32>
// CHECK-LE: extractelement <4 x i32>

  res_f  = vec_extract(vf, param_i);
// CHECK: extractelement <4 x float>
// CHECK-LE: extractelement <4 x float>

  /* vec_insert */
  res_vsc = vec_insert(param_sc, vsc, param_i);
// CHECK: insertelement <16 x i8>
// CHECK-LE: insertelement <16 x i8>

  res_vuc = vec_insert(param_uc, vuc, param_i);
// CHECK: insertelement <16 x i8>
// CHECK-LE: insertelement <16 x i8>

  res_vbc = vec_insert(param_uc, vbc, param_i);
// CHECK: insertelement <16 x i8>
// CHECK-LE: insertelement <16 x i8>

  res_vs  = vec_insert(param_s, vs, param_i);
// CHECK: insertelement <8 x i16>
// CHECK-LE: insertelement <8 x i16>

  res_vus = vec_insert(param_us, vus, param_i);
// CHECK: insertelement <8 x i16>
// CHECK-LE: insertelement <8 x i16>

  res_vbs = vec_insert(param_us, vbs, param_i);
// CHECK: insertelement <8 x i16>
// CHECK-LE: insertelement <8 x i16>

  res_vi  = vec_insert(param_i, vi, param_i);
// CHECK: insertelement <4 x i32>
// CHECK-LE: insertelement <4 x i32>

  res_vui = vec_insert(param_ui, vui, param_i);
// CHECK: insertelement <4 x i32>
// CHECK-LE: insertelement <4 x i32>

  res_vbi = vec_insert(param_ui, vbi, param_i);
// CHECK: insertelement <4 x i32>
// CHECK-LE: insertelement <4 x i32>

  res_vf  = vec_insert(param_f, vf, param_i);
// CHECK: insertelement <4 x float>
// CHECK-LE: insertelement <4 x float>

  /* vec_lvlx */
  res_vsc = vec_lvlx(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_lvlx(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvlx(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvlx(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_lvlx(0, &vbc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvlx(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvlx(0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvlx(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvlx(0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_lvlx(0, &vbs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_lvlx(0, &vp);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvlx(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvlx(0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvlx(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvlx(0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_lvlx(0, &vbi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_lvlx(0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_lvlxl */
  res_vsc = vec_lvlxl(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_lvlxl(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvlxl(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvlxl(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_lvlxl(0, &vbc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvlxl(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvlxl(0, &vs);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvlxl(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvlxl(0, &vus);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_lvlxl(0, &vbs);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_lvlxl(0, &vp);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvlxl(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvlxl(0, &vi);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvlxl(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvlxl(0, &vui);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_lvlxl(0, &vbi);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_lvlxl(0, &vf);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_lvrx */
  res_vsc = vec_lvrx(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_lvrx(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvrx(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvrx(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_lvrx(0, &vbc);
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvrx(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvrx(0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvrx(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvrx(0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_lvrx(0, &vbs);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_lvrx(0, &vp);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvrx(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvrx(0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvrx(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvrx(0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_lvrx(0, &vbi);
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_lvrx(0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_lvrxl */
  res_vsc = vec_lvrxl(0, param_sc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vsc = vec_lvrxl(0, &vsc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvrxl(0, param_uc_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vuc = vec_lvrxl(0, &vuc);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbc = vec_lvrxl(0, &vbc);
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvrxl(0, param_s_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vs  = vec_lvrxl(0, &vs);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvrxl(0, param_us_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vus = vec_lvrxl(0, &vus);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbs = vec_lvrxl(0, &vbs);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vp  = vec_lvrxl(0, &vp);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvrxl(0, param_i_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vi  = vec_lvrxl(0, &vi);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvrxl(0, param_ui_ld);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vui = vec_lvrxl(0, &vui);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vbi = vec_lvrxl(0, &vbi);
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm

  res_vf  = vec_lvrxl(0, &vf);
// CHECK: @llvm.ppc.altivec.lvxl
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvxl
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm

  /* vec_stvlx */
  vec_stvlx(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vbc, 0, &vbc);
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vbs, 0, &vbs);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vp, 0, &vp);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vbi, 0, &vbi);
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvlx(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  /* vec_stvlxl */
  vec_stvlxl(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vbc, 0, &vbc);
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vbs, 0, &vbs);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vp, 0, &vp);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vbi, 0, &vbi);
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvlxl(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  /* vec_stvrx */
  vec_stvrx(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vbc, 0, &vbc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vbs, 0, &vbs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vp, 0, &vp);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vbi, 0, &vbi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  vec_stvrx(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvx
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvx

  /* vec_stvrxl */
  vec_stvrxl(vsc, 0, &param_sc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vsc, 0, &vsc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vuc, 0, &param_uc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vuc, 0, &vuc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vbc, 0, &vbc);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <16 x i8> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vs, 0, &param_s);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vs, 0, &vs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vus, 0, &param_us);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vus, 0, &vus);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vbs, 0, &vbs);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vp, 0, &vp);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <8 x i16> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vi, 0, &param_i);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vi, 0, &vi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vui, 0, &param_ui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vui, 0, &vui);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vbi, 0, &vbi);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: store <4 x i32> zeroinitializer
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  vec_stvrxl(vf, 0, &vf);
// CHECK: @llvm.ppc.altivec.lvx
// CHECK: @llvm.ppc.altivec.lvsl
// CHECK: store <4 x float> zeroinitializer
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.lvsr
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.altivec.stvxl
// CHECK-LE: @llvm.ppc.altivec.lvx
// CHECK-LE: @llvm.ppc.altivec.lvsl
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.lvsr
// CHECK-LE: @llvm.ppc.altivec.vperm
// CHECK-LE: @llvm.ppc.altivec.stvxl

  /* vec_promote */
  res_vsc = vec_promote(param_sc, 0);
// CHECK: store <16 x i8> zeroinitializer
// CHECK: insertelement <16 x i8>
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: insertelement <16 x i8>

  res_vuc = vec_promote(param_uc, 0);
// CHECK: store <16 x i8> zeroinitializer
// CHECK: insertelement <16 x i8>
// CHECK-LE: store <16 x i8> zeroinitializer
// CHECK-LE: insertelement <16 x i8>

  res_vs  = vec_promote(param_s, 0);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: insertelement <8 x i16>
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: insertelement <8 x i16>

  res_vus = vec_promote(param_us, 0);
// CHECK: store <8 x i16> zeroinitializer
// CHECK: insertelement <8 x i16>
// CHECK-LE: store <8 x i16> zeroinitializer
// CHECK-LE: insertelement <8 x i16>

  res_vi  = vec_promote(param_i, 0);
// CHECK: store <4 x i32> zeroinitializer
// CHECK: insertelement <4 x i32>
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: insertelement <4 x i32>

  res_vui = vec_promote(param_ui, 0);
// CHECK: store <4 x i32> zeroinitializer
// CHECK: insertelement <4 x i32>
// CHECK-LE: store <4 x i32> zeroinitializer
// CHECK-LE: insertelement <4 x i32>

  res_vf  = vec_promote(param_f, 0);
// CHECK: store <4 x float> zeroinitializer
// CHECK: insertelement <4 x float>
// CHECK-LE: store <4 x float> zeroinitializer
// CHECK-LE: insertelement <4 x float>

  /* vec_splats */
  res_vsc = vec_splats(param_sc);
// CHECK: insertelement <16 x i8>
// CHECK-LE: insertelement <16 x i8>

  res_vuc = vec_splats(param_uc);
// CHECK: insertelement <16 x i8>
// CHECK-LE: insertelement <16 x i8>

  res_vs  = vec_splats(param_s);
// CHECK: insertelement <8 x i16>
// CHECK-LE: insertelement <8 x i16>

  res_vus = vec_splats(param_us);
// CHECK: insertelement <8 x i16>
// CHECK-LE: insertelement <8 x i16>

  res_vi  = vec_splats(param_i);
// CHECK: insertelement <4 x i32>
// CHECK-LE: insertelement <4 x i32>

  res_vui = vec_splats(param_ui);
// CHECK: insertelement <4 x i32>
// CHECK-LE: insertelement <4 x i32>

  res_vf  = vec_splats(param_f);
// CHECK: insertelement <4 x float>
// CHECK-LE: insertelement <4 x float>

  /* ------------------------------ predicates -------------------------------------- */

  /* vec_all_eq */
  res_i = vec_all_eq(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_eq(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_eq(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_eq(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_eq(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_eq(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_eq(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_eq(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vp, vp);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_eq(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_eq(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_eq(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_eq(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_eq(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_eq(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_eq(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_eq(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_all_ge */
  res_i = vec_all_ge(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_ge(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_ge(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_ge(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_ge(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_ge(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_ge(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_ge(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_ge(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_ge(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_ge(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_ge(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_ge(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_ge(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_ge(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_ge(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_ge(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_ge(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_ge(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_ge(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_ge(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_ge(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_all_gt */
  res_i = vec_all_gt(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_gt(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_gt(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_gt(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_gt(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_gt(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_gt(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_gt(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_gt(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_gt(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_gt(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_gt(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_gt(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_gt(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_gt(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_gt(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_gt(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_gt(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_gt(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_gt(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_gt(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_gt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_in */
  res_i = vec_all_in(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpbfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpbfp.p

  /* vec_all_le */
  res_i = vec_all_le(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_le(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_le(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_le(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_le(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_le(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_le(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_le(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_le(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_le(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_le(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_le(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_le(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_le(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_le(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_le(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_le(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_le(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_le(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_le(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_le(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_le(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_all_lt */
  res_i = vec_all_lt(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_lt(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_lt(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_lt(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_lt(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_all_lt(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_lt(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_all_lt(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_lt(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_lt(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_lt(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_lt(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_all_lt(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_lt(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_all_lt(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_lt(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_lt(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_lt(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_lt(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_all_lt(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_lt(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_all_lt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_nan */
  res_i = vec_all_nan(vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /*  vec_all_ne */
  res_i = vec_all_ne(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_ne(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_ne(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_ne(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_ne(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_ne(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_ne(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_all_ne(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vp, vp);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_all_ne(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_ne(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_ne(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_ne(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_ne(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_ne(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_ne(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_all_ne(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_all_nge */
  res_i = vec_all_nge(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_all_ngt */
  res_i = vec_all_ngt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_nle */
  res_i = vec_all_nle(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_all_nlt */
  res_i = vec_all_nlt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_numeric */
  res_i = vec_all_numeric(vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /*  vec_any_eq */
  res_i = vec_any_eq(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_eq(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_eq(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_eq(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_eq(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_eq(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_eq(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_eq(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vp, vp);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_eq(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_eq(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_eq(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_eq(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_eq(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_eq(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_eq(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_eq(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_ge */
  res_i = vec_any_ge(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_ge(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_ge(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_ge(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_ge(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_ge(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_ge(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_ge(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_ge(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_ge(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_ge(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_ge(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_ge(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_ge(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_ge(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_ge(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_ge(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_ge(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_ge(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_ge(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_ge(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_ge(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_any_gt */
  res_i = vec_any_gt(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_gt(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_gt(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_gt(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_gt(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_gt(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_gt(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_gt(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_gt(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_gt(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_gt(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_gt(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_gt(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_gt(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_gt(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_gt(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_gt(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_gt(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_gt(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_gt(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_gt(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_gt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_le */
  res_i = vec_any_le(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_le(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_le(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_le(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_le(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_le(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_le(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_le(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_le(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_le(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_le(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_le(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_le(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_le(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_le(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_le(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_le(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_le(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_le(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_le(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_le(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_le(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_any_lt */
  res_i = vec_any_lt(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_lt(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_lt(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_lt(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_lt(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p

  res_i = vec_any_lt(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_lt(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p

  res_i = vec_any_lt(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_lt(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_lt(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_lt(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_lt(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p

  res_i = vec_any_lt(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_lt(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p

  res_i = vec_any_lt(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_lt(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_lt(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_lt(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_lt(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p

  res_i = vec_any_lt(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_lt(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p

  res_i = vec_any_lt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_nan */
  res_i = vec_any_nan(vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_ne */
  res_i = vec_any_ne(vsc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_ne(vsc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_ne(vuc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_ne(vuc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_ne(vbc, vsc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_ne(vbc, vuc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_ne(vbc, vbc);
// CHECK: @llvm.ppc.altivec.vcmpequb.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p

  res_i = vec_any_ne(vs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vus, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vus, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vbs, vs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vbs, vus);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vbs, vbs);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vp, vp);
// CHECK: @llvm.ppc.altivec.vcmpequh.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p

  res_i = vec_any_ne(vi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_ne(vi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_ne(vui, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_ne(vui, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_ne(vbi, vi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_ne(vbi, vui);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_ne(vbi, vbi);
// CHECK: @llvm.ppc.altivec.vcmpequw.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p

  res_i = vec_any_ne(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_nge */
  res_i = vec_any_nge(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_any_ngt */
  res_i = vec_any_ngt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_nle */
  res_i = vec_any_nle(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_any_nlt */
  res_i = vec_any_nlt(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_numeric */
  res_i = vec_any_numeric(vf);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_out */
  res_i = vec_any_out(vf, vf);
// CHECK: @llvm.ppc.altivec.vcmpbfp.p
// CHECK-LE: @llvm.ppc.altivec.vcmpbfp.p
}

/* ------------------------------ Relational Operators ------------------------------ */
// CHECK-LABEL: define{{.*}} void @test7
void test7() {
  vector signed char vsc1 = (vector signed char)(-1);
  vector signed char vsc2 = (vector signed char)(-2);
  res_i = (vsc1 == vsc2);
// CHECK: @llvm.ppc.altivec.vcmpequb.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p(i32 2

  res_i = (vsc1 != vsc2);
// CHECK: @llvm.ppc.altivec.vcmpequb.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p(i32 0

  res_i = (vsc1 <  vsc2);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p(i32 2

  res_i = (vsc1 >  vsc2);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p(i32 2

  res_i = (vsc1 <= vsc2);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p(i32 0

  res_i = (vsc1 >= vsc2);
// CHECK: @llvm.ppc.altivec.vcmpgtsb.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsb.p(i32 0

  vector unsigned char vuc1 = (vector unsigned char)(1);
  vector unsigned char vuc2 = (vector unsigned char)(2);
  res_i = (vuc1 == vuc2);
// CHECK: @llvm.ppc.altivec.vcmpequb.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p(i32 2

  res_i = (vuc1 != vuc2);
// CHECK: @llvm.ppc.altivec.vcmpequb.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpequb.p(i32 0

  res_i = (vuc1 <  vuc2);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p(i32 2

  res_i = (vuc1 >  vuc2);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p(i32 2

  res_i = (vuc1 <= vuc2);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p(i32 0

  res_i = (vuc1 >= vuc2);
// CHECK: @llvm.ppc.altivec.vcmpgtub.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtub.p(i32 0

  vector short vs1 = (vector short)(-1);
  vector short vs2 = (vector short)(-2);
  res_i = (vs1 == vs2);
// CHECK: @llvm.ppc.altivec.vcmpequh.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p(i32 2

  res_i = (vs1 != vs2);
// CHECK: @llvm.ppc.altivec.vcmpequh.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p(i32 0

  res_i = (vs1 <  vs2);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p(i32 2

  res_i = (vs1 >  vs2);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p(i32 2

  res_i = (vs1 <= vs2);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p(i32 0

  res_i = (vs1 >= vs2);
// CHECK: @llvm.ppc.altivec.vcmpgtsh.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsh.p(i32 0

  vector unsigned short vus1 = (vector unsigned short)(1);
  vector unsigned short vus2 = (vector unsigned short)(2);
  res_i = (vus1 == vus2);
// CHECK: @llvm.ppc.altivec.vcmpequh.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p(i32 2

  res_i = (vus1 != vus2);
// CHECK: @llvm.ppc.altivec.vcmpequh.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpequh.p(i32 0

  res_i = (vus1 <  vus2);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p(i32 2

  res_i = (vus1 >  vus2);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p(i32 2

  res_i = (vus1 <= vus2);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p(i32 0

  res_i = (vus1 >= vus2);
// CHECK: @llvm.ppc.altivec.vcmpgtuh.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuh.p(i32 0

  vector int vi1 = (vector int)(-1);
  vector int vi2 = (vector int)(-2);
  res_i = (vi1 == vi2);
// CHECK: @llvm.ppc.altivec.vcmpequw.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p(i32 2

  res_i = (vi1 != vi2);
// CHECK: @llvm.ppc.altivec.vcmpequw.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p(i32 0

  res_i = (vi1 <  vi2);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p(i32 2

  res_i = (vi1 >  vi2);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p(i32 2

  res_i = (vi1 <= vi2);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p(i32 0

  res_i = (vi1 >= vi2);
// CHECK: @llvm.ppc.altivec.vcmpgtsw.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsw.p(i32 0

  vector unsigned int vui1 = (vector unsigned int)(1);
  vector unsigned int vui2 = (vector unsigned int)(2);
  res_i = (vui1 == vui2);
// CHECK: @llvm.ppc.altivec.vcmpequw.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p(i32 2

  res_i = (vui1 != vui2);
// CHECK: @llvm.ppc.altivec.vcmpequw.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpequw.p(i32 0

  res_i = (vui1 <  vui2);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p(i32 2

  res_i = (vui1 >  vui2);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p(i32 2

  res_i = (vui1 <= vui2);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p(i32 0

  res_i = (vui1 >= vui2);
// CHECK: @llvm.ppc.altivec.vcmpgtuw.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpgtuw.p(i32 0

  vector float vf1 = (vector float)(1.0);
  vector float vf2 = (vector float)(2.0);
  res_i = (vf1 == vf2);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p(i32 2

  res_i = (vf1 != vf2);
// CHECK: @llvm.ppc.altivec.vcmpeqfp.p(i32 0
// CHECK-LE: @llvm.ppc.altivec.vcmpeqfp.p(i32 0

  res_i = (vf1 <  vf2);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p(i32 2

  res_i = (vf1 >  vf2);
// CHECK: @llvm.ppc.altivec.vcmpgtfp.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgtfp.p(i32 2

  res_i = (vf1 <= vf2);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p(i32 2

  res_i = (vf1 >= vf2);
// CHECK: @llvm.ppc.altivec.vcmpgefp.p(i32 2
// CHECK-LE: @llvm.ppc.altivec.vcmpgefp.p(i32 2
}

/* ------------------------------ optional ---------------------------------- */
void test8() {
// CHECK-LABEL: define{{.*}} void @test8
// CHECK-LE-LABEL: define{{.*}} void @test8
  res_vbc = vec_reve(vbc);
  // CHECK: shufflevector <16 x i8> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <16 x i8> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  res_vsc = vec_reve(vsc);
  // CHECK: shufflevector <16 x i8> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <16 x i8> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  res_vuc = vec_reve(vuc);
  // CHECK: shufflevector <16 x i8> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <16 x i8> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  res_vbi = vec_reve(vbi);
  // CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>

  res_vi = vec_reve(vi);
  // CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>

  res_vui = vec_reve(vui);
  // CHECK: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <4 x i32> %{{[0-9]+}}, <4 x i32> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>

  res_vbs = vec_reve(vbs);
  // CHECK: shufflevector <8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  res_vs = vec_reve(vs);
  // CHECK: shufflevector <8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  res_vus = vec_reve(vus);
  // CHECK: shufflevector <8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <8 x i16> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  res_vf = vec_reve(vf);
  // CHECK: shufflevector <4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK-LE: shufflevector <4 x float> %{{[0-9]+}}, <4 x float> %{{[0-9]+}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>

  res_vbc = vec_revb(vbc);
// CHECK: [[T1:%.+]] = load <16 x i8>, ptr @vbc, align 16
// CHECK: store <16 x i8> [[T1]], ptr [[T2:%.+]], align 16
// CHECK: [[T3:%.+]] = load <16 x i8>, ptr [[T2]], align 16
// CHECK: store <16 x i8> [[T3]], ptr @res_vbc, align 16
// CHECK-LE: [[T1:%.+]] = load <16 x i8>, ptr @vbc, align 16
// CHECK-LE: store <16 x i8> [[T1]], ptr [[T2:%.+]], align 16
// CHECK-LE: [[T3:%.+]] = load <16 x i8>, ptr [[T2]], align 16
// CHECK-LE: store <16 x i8> [[T3]], ptr @res_vbc, align 16

  res_vsc = vec_revb(vsc);
// CHECK: [[T1:%.+]] = load <16 x i8>, ptr @vsc, align 16
// CHECK: store <16 x i8> [[T1]], ptr [[T2:%.+]], align 16
// CHECK: [[T3:%.+]] = load <16 x i8>, ptr [[T2]], align 16
// CHECK: store <16 x i8> [[T3]], ptr @res_vsc, align 16
// CHECK-LE: [[T1:%.+]] = load <16 x i8>, ptr @vsc, align 16
// CHECK-LE: store <16 x i8> [[T1]], ptr [[T2:%.+]], align 16
// CHECK-LE: [[T3:%.+]] = load <16 x i8>, ptr [[T2]], align 16
// CHECK-LE: store <16 x i8> [[T3]], ptr @res_vsc, align 16

  res_vuc = vec_revb(vuc);
// CHECK: [[T1:%.+]] = load <16 x i8>, ptr @vuc, align 16
// CHECK: store <16 x i8> [[T1]], ptr [[T2:%.+]], align 16
// CHECK: [[T3:%.+]] = load <16 x i8>, ptr [[T2]], align 16
// CHECK: store <16 x i8> [[T3]], ptr @res_vuc, align 16
// CHECK-LE: [[T1:%.+]] = load <16 x i8>, ptr @vuc, align 16
// CHECK-LE: store <16 x i8> [[T1]], ptr [[T2:%.+]], align 16
// CHECK-LE: [[T3:%.+]] = load <16 x i8>, ptr [[T2]], align 16
// CHECK-LE: store <16 x i8> [[T3]], ptr @res_vuc, align 16

  res_vbs = vec_revb(vbs);
// CHECK: store <16 x i8> <i8 1, i8 0, i8 3, i8 2, i8 5, i8 4, i8 7, i8 6, i8 9, i8 8, i8 11, i8 10, i8 13, i8 12, i8 15, i8 14>, ptr {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 1, i8 0, i8 3, i8 2, i8 5, i8 4, i8 7, i8 6, i8 9, i8 8, i8 11, i8 10, i8 13, i8 12, i8 15, i8 14>, ptr {{%.+}}, align 16
// CHECK-LE: store <16 x i8> splat (i8 -1), ptr {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vs = vec_revb(vs);
// CHECK: store <16 x i8> <i8 1, i8 0, i8 3, i8 2, i8 5, i8 4, i8 7, i8 6, i8 9, i8 8, i8 11, i8 10, i8 13, i8 12, i8 15, i8 14>, ptr {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 1, i8 0, i8 3, i8 2, i8 5, i8 4, i8 7, i8 6, i8 9, i8 8, i8 11, i8 10, i8 13, i8 12, i8 15, i8 14>, ptr {{%.+}}, align 16
// CHECK-LE: store <16 x i8> splat (i8 -1), ptr {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vus = vec_revb(vus);
// CHECK: store <16 x i8> <i8 1, i8 0, i8 3, i8 2, i8 5, i8 4, i8 7, i8 6, i8 9, i8 8, i8 11, i8 10, i8 13, i8 12, i8 15, i8 14>, ptr {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 1, i8 0, i8 3, i8 2, i8 5, i8 4, i8 7, i8 6, i8 9, i8 8, i8 11, i8 10, i8 13, i8 12, i8 15, i8 14>, ptr {{%.+}}, align 16
// CHECK-LE: store <16 x i8> splat (i8 -1), ptr {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vbi = vec_revb(vbi);
// CHECK: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK-LE: store <16 x i8> splat (i8 -1), ptr {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vi = vec_revb(vi);
// CHECK: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK-LE: store <16 x i8> splat (i8 -1), ptr {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vui = vec_revb(vui);
// CHECK: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK-LE: store <16 x i8> splat (i8 -1), ptr {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})

  res_vf = vec_revb(vf);
// CHECK: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
// CHECK-LE: store <16 x i8> <i8 3, i8 2, i8 1, i8 0, i8 7, i8 6, i8 5, i8 4, i8 11, i8 10, i8 9, i8 8, i8 15, i8 14, i8 13, i8 12>, ptr {{%.+}}, align 16
// CHECK-LE: store <16 x i8> splat (i8 -1), ptr {{%.+}}, align 16
// CHECK-LE: xor <16 x i8>
// CHECK-LE: call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> {{%.+}}, <4 x i32> {{%.+}}, <16 x i8> {{%.+}})
}

/* ------------------------------ vec_xl ------------------------------------ */
void test9() {
  // CHECK-LABEL: define{{.*}} void @test9
  // CHECK-LE-LABEL: define{{.*}} void @test9
  res_vsc = vec_xl(param_sll, param_sc_ld);
  // CHECK: load <16 x i8>, ptr %{{.+}}, align 1
  // CHECK-LE: load <16 x i8>, ptr %{{.+}}, align 1

  res_vuc = vec_xl(param_sll, param_uc_ld);
  // CHECK: load <16 x i8>, ptr %{{.+}}, align 1
  // CHECK-LE: load <16 x i8>, ptr %{{.+}}, align 1

  res_vs = vec_xl(param_sll, param_s_ld);
  // CHECK: load <8 x i16>, ptr %{{.+}}, align 1
  // CHECK-LE: load <8 x i16>, ptr %{{.+}}, align 1

  res_vus = vec_xl(param_sll, param_us_ld);
  // CHECK: load <8 x i16>, ptr %{{.+}}, align 1
  // CHECK-LE: load <8 x i16>, ptr %{{.+}}, align 1

  res_vi = vec_xl(param_sll, param_i_ld);
  // CHECK: load <4 x i32>, ptr %{{.+}}, align 1
  // CHECK-LE: load <4 x i32>, ptr %{{.+}}, align 1

  res_vui = vec_xl(param_sll, param_ui_ld);
  // CHECK: load <4 x i32>, ptr %{{.+}}, align 1
  // CHECK-LE: load <4 x i32>, ptr %{{.+}}, align 1

  res_vf = vec_xl(param_sll, param_f_ld);
  // CHECK: load <4 x float>, ptr %{{.+}}, align 1
  // CHECK-LE: load <4 x float>, ptr %{{.+}}, align 1
}

/* ------------------------------ vec_xst ----------------------------------- */
void test10() {
  // CHECK-LABEL: define{{.*}} void @test10
  // CHECK-LE-LABEL: define{{.*}} void @test10
  vec_xst(vsc, param_sll, &param_sc);
  // CHECK: store <16 x i8> %{{[0-9]+}}, ptr %{{.+}}, align 1
  // CHECK-LE: store <16 x i8> %{{[0-9]+}}, ptr %{{.+}}, align 1

  vec_xst(vuc, param_sll, &param_uc);
  // CHECK: store <16 x i8> %{{[0-9]+}}, ptr %{{.+}}, align 1
  // CHECK-LE: store <16 x i8> %{{[0-9]+}}, ptr %{{.+}}, align 1

  vec_xst(vs, param_sll, &param_s);
  // CHECK: store <8 x i16> %{{[0-9]+}}, ptr %{{.+}}, align 1
  // CHECK-LE: store <8 x i16> %{{[0-9]+}}, ptr %{{.+}}, align 1

  vec_xst(vus, param_sll, &param_us);
  // CHECK: store <8 x i16> %{{[0-9]+}}, ptr %{{.+}}, align 1
  // CHECK-LE: store <8 x i16> %{{[0-9]+}}, ptr %{{.+}}, align 1

  vec_xst(vi, param_sll, &param_i);
  // CHECK: store <4 x i32> %{{[0-9]+}}, ptr %{{.+}}, align 1
  // CHECK-LE: store <4 x i32> %{{[0-9]+}}, ptr %{{.+}}, align 1

  vec_xst(vui, param_sll, &param_ui);
  // CHECK: store <4 x i32> %{{[0-9]+}}, ptr %{{.+}}, align 1
  // CHECK-LE: store <4 x i32> %{{[0-9]+}}, ptr %{{.+}}, align 1

  vec_xst(vf, param_sll, &param_f);
  // CHECK: store <4 x float> %{{[0-9]+}}, ptr %{{.+}}, align 1
  // CHECK-LE: store <4 x float> %{{[0-9]+}}, ptr %{{.+}}, align 1
}

vector float test_rsqrtf(vector float a, vector float b) {
  // CHECK-LABEL: test_rsqrtf
  // CHECK: call fast <4 x float> @llvm.sqrt.v4f32
  // CHECK: fdiv fast <4 x float> splat (float 1.000000e+00)
  // CHECK-LE-LABEL: test_rsqrtf
  // CHECK-LE: call fast <4 x float> @llvm.sqrt.v4f32
  // CHECK-LE: fdiv fast <4 x float> splat (float 1.000000e+00)
  return vec_rsqrt(a);
}

vector float test_recipdivf(vector float a, vector float b) {
  // CHECK-LABEL: test_recipdivf
  // CHECK: fdiv fast <4 x float>
  // CHECK-LE-LABEL: test_recipdivf
  // CHECK-LE: fdiv fast <4 x float>
  return vec_recipdiv(a, b);
}
