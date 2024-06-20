// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed char vsc;
volatile vector signed short vss;
volatile vector signed int vsi;
volatile vector signed long long vsl;
volatile vector unsigned char vuc;
volatile vector unsigned short vus;
volatile vector unsigned int vui;
volatile vector unsigned long long vul;
volatile vector bool char vbc;
volatile vector bool short vbs;
volatile vector bool int vbi;
volatile vector bool long long vbl;
volatile vector double vd;

volatile signed char sc;
volatile signed short ss;
volatile signed int si;
volatile signed long long sl;
volatile unsigned char uc;
volatile unsigned short us;
volatile unsigned int ui;
volatile unsigned long long ul;
volatile double d;

const void * volatile cptr;
const signed char * volatile cptrsc;
const signed short * volatile cptrss;
const signed int * volatile cptrsi;
const signed long long * volatile cptrsl;
const unsigned char * volatile cptruc;
const unsigned short * volatile cptrus;
const unsigned int * volatile cptrui;
const unsigned long long * volatile cptrul;
const float * volatile cptrf;
const double * volatile cptrd;

void * volatile ptr;
signed char * volatile ptrsc;
signed short * volatile ptrss;
signed int * volatile ptrsi;
signed long long * volatile ptrsl;
unsigned char * volatile ptruc;
unsigned short * volatile ptrus;
unsigned int * volatile ptrui;
unsigned long long * volatile ptrul;
float * volatile ptrf;
double * volatile ptrd;

volatile unsigned int len;
volatile int idx;
int cc;

void test_core(void) {
  // CHECK-ASM-LABEL: test_core

  len = __lcbb(cptr, 64);
  // CHECK: call i32 @llvm.s390.lcbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: lcbb
  len = __lcbb(cptr, 128);
  // CHECK: call i32 @llvm.s390.lcbb(ptr %{{.*}}, i32 1)
  // CHECK-ASM: lcbb
  len = __lcbb(cptr, 256);
  // CHECK: call i32 @llvm.s390.lcbb(ptr %{{.*}}, i32 2)
  // CHECK-ASM: lcbb
  len = __lcbb(cptr, 512);
  // CHECK: call i32 @llvm.s390.lcbb(ptr %{{.*}}, i32 3)
  // CHECK-ASM: lcbb
  len = __lcbb(cptr, 1024);
  // CHECK: call i32 @llvm.s390.lcbb(ptr %{{.*}}, i32 4)
  // CHECK-ASM: lcbb
  len = __lcbb(cptr, 2048);
  // CHECK: call i32 @llvm.s390.lcbb(ptr %{{.*}}, i32 5)
  // CHECK-ASM: lcbb
  len = __lcbb(cptr, 4096);
  // CHECK: call i32 @llvm.s390.lcbb(ptr %{{.*}}, i32 6)
  // CHECK-ASM: lcbb

  sc = vec_extract(vsc, idx);
  // CHECK: extractelement <16 x i8> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvb
  uc = vec_extract(vuc, idx);
  // CHECK: extractelement <16 x i8> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvb
  uc = vec_extract(vbc, idx);
  // CHECK: extractelement <16 x i8> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvb
  ss = vec_extract(vss, idx);
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvh
  us = vec_extract(vus, idx);
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvh
  us = vec_extract(vbs, idx);
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvh
  si = vec_extract(vsi, idx);
  // CHECK: extractelement <4 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvf
  ui = vec_extract(vui, idx);
  // CHECK: extractelement <4 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvf
  ui = vec_extract(vbi, idx);
  // CHECK: extractelement <4 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvf
  sl = vec_extract(vsl, idx);
  // CHECK: extractelement <2 x i64> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvg
  ul = vec_extract(vul, idx);
  // CHECK: extractelement <2 x i64> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvg
  ul = vec_extract(vbl, idx);
  // CHECK: extractelement <2 x i64> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvg
  d = vec_extract(vd, idx);
  // CHECK: extractelement <2 x double> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvg

  vsc = vec_insert(sc, vsc, idx);
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgb
  vuc = vec_insert(uc, vuc, idx);
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgb
  vuc = vec_insert(uc, vbc, idx);
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgb
  vss = vec_insert(ss, vss, idx);
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgh
  vus = vec_insert(us, vus, idx);
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgh
  vus = vec_insert(us, vbs, idx);
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgh
  vsi = vec_insert(si, vsi, idx);
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgf
  vui = vec_insert(ui, vui, idx);
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgf
  vui = vec_insert(ui, vbi, idx);
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgf
  vsl = vec_insert(sl, vsl, idx);
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg
  vul = vec_insert(ul, vul, idx);
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg
  vul = vec_insert(ul, vbl, idx);
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg
  vd = vec_insert(d, vd, idx);
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg

  vsc = vec_promote(sc, idx);
  // CHECK: insertelement <16 x i8> poison, i8 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgb
  vuc = vec_promote(uc, idx);
  // CHECK: insertelement <16 x i8> poison, i8 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgb
  vss = vec_promote(ss, idx);
  // CHECK: insertelement <8 x i16> poison, i16 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgh
  vus = vec_promote(us, idx);
  // CHECK: insertelement <8 x i16> poison, i16 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgh
  vsi = vec_promote(si, idx);
  // CHECK: insertelement <4 x i32> poison, i32 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgf
  vui = vec_promote(ui, idx);
  // CHECK: insertelement <4 x i32> poison, i32 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgf
  vsl = vec_promote(sl, idx);
  // CHECK: insertelement <2 x i64> poison, i64 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg
  vul = vec_promote(ul, idx);
  // CHECK: insertelement <2 x i64> poison, i64 %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg
  vd = vec_promote(d, idx);
  // CHECK: insertelement <2 x double> poison, double %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg

  vsc = vec_insert_and_zero(cptrsc);
  // CHECK: insertelement <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 poison, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %{{.*}}, i64 7
  // CHECK-ASM: vllezb
  vuc = vec_insert_and_zero(cptruc);
  // CHECK: insertelement <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 poison, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %{{.*}}, i64 7
  // CHECK-ASM: vllezb
  vss = vec_insert_and_zero(cptrss);
  // CHECK: insertelement <8 x i16> <i16 0, i16 0, i16 0, i16 poison, i16 0, i16 0, i16 0, i16 0>, i16 %{{.*}}, i64 3
  // CHECK-ASM: vllezh
  vus = vec_insert_and_zero(cptrus);
  // CHECK: insertelement <8 x i16> <i16 0, i16 0, i16 0, i16 poison, i16 0, i16 0, i16 0, i16 0>, i16 %{{.*}}, i64 3
  // CHECK-ASM: vllezh
  vsi = vec_insert_and_zero(cptrsi);
  // CHECK: insertelement <4 x i32> <i32 0, i32 poison, i32 0, i32 0>, i32 %{{.*}}, i64 1
  // CHECK-ASM: vllezf
  vui = vec_insert_and_zero(cptrui);
  // CHECK: insertelement <4 x i32> <i32 0, i32 poison, i32 0, i32 0>, i32 %{{.*}}, i64 1
  // CHECK-ASM: vllezf
  vsl = vec_insert_and_zero(cptrsl);
  // CHECK: insertelement <2 x i64> <i64 poison, i64 0>, i64 %{{.*}}, i64 0
  // CHECK-ASM: vllezg
  vul = vec_insert_and_zero(cptrul);
  // CHECK: insertelement <2 x i64> <i64 poison, i64 0>, i64 %{{.*}}, i64 0
  // CHECK-ASM: vllezg
  vd = vec_insert_and_zero(cptrd);
  // CHECK: insertelement <2 x double> <double poison, double 0.000000e+00>, double %{{.*}}, i64 0
  // CHECK-ASM: vllezg

  vsc = vec_perm(vsc, vsc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vuc = vec_perm(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vbc = vec_perm(vbc, vbc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vss = vec_perm(vss, vss, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vus = vec_perm(vus, vus, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vbs = vec_perm(vbs, vbs, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vsi = vec_perm(vsi, vsi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vui = vec_perm(vui, vui, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vbi = vec_perm(vbi, vbi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vsl = vec_perm(vsl, vsl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vul = vec_perm(vul, vul, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vbl = vec_perm(vbl, vbl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vd = vec_perm(vd, vd, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm

  vsl = vec_permi(vsl, vsl, 0);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  // CHECK-ASM: vpdi
  vsl = vec_permi(vsl, vsl, 1);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 1)
  // CHECK-ASM: vpdi
  vsl = vec_permi(vsl, vsl, 2);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 4)
  // CHECK-ASM: vpdi
  vsl = vec_permi(vsl, vsl, 3);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 5)
  // CHECK-ASM: vpdi
  vul = vec_permi(vul, vul, 0);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  // CHECK-ASM: vpdi
  vul = vec_permi(vul, vul, 1);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 1)
  // CHECK-ASM: vpdi
  vul = vec_permi(vul, vul, 2);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 4)
  // CHECK-ASM: vpdi
  vul = vec_permi(vul, vul, 3);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 5)
  // CHECK-ASM: vpdi
  vbl = vec_permi(vbl, vbl, 0);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  // CHECK-ASM: vpdi
  vbl = vec_permi(vbl, vbl, 1);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 1)
  // CHECK-ASM: vpdi
  vbl = vec_permi(vbl, vbl, 2);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 4)
  // CHECK-ASM: vpdi
  vbl = vec_permi(vbl, vbl, 3);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 5)
  // CHECK-ASM: vpdi
  vd = vec_permi(vd, vd, 0);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  // CHECK-ASM: vpdi
  vd = vec_permi(vd, vd, 1);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 1)
  // CHECK-ASM: vpdi
  vd = vec_permi(vd, vd, 2);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 4)
  // CHECK-ASM: vpdi
  vd = vec_permi(vd, vd, 3);
  // CHECK: call <2 x i64> @llvm.s390.vpdi(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 5)
  // CHECK-ASM: vpdi

  vss = vec_revb(vss);
  // CHECK-ASM: vperm
  vus = vec_revb(vus);
  // CHECK-ASM: vperm
  vsi = vec_revb(vsi);
  // CHECK-ASM: vperm
  vui = vec_revb(vui);
  // CHECK-ASM: vperm
  vsl = vec_revb(vsl);
  // CHECK-ASM: vperm
  vul = vec_revb(vul);
  // CHECK-ASM: vperm
  vd = vec_revb(vd);
  // CHECK-ASM: vperm

  vsc = vec_reve(vsc);
  // CHECK-ASM: vperm
  vuc = vec_reve(vuc);
  // CHECK-ASM: vperm
  vbc = vec_reve(vbc);
  // CHECK-ASM: vperm
  vss = vec_reve(vss);
  // CHECK-ASM: vperm
  vus = vec_reve(vus);
  // CHECK-ASM: vperm
  vbs = vec_reve(vbs);
  // CHECK-ASM: vperm
  vsi = vec_reve(vsi);
  // CHECK-ASM: vperm
  vui = vec_reve(vui);
  // CHECK-ASM: vperm
  vbi = vec_reve(vbi);
  // CHECK-ASM: vperm
  vsl = vec_reve(vsl);
  // CHECK-ASM: {{vperm|vpdi}}
  vul = vec_reve(vul);
  // CHECK-ASM: {{vperm|vpdi}}
  vbl = vec_reve(vbl);
  // CHECK-ASM: {{vperm|vpdi}}
  vd = vec_reve(vd);
  // CHECK-ASM: {{vperm|vpdi}}

  vsc = vec_sel(vsc, vsc, vuc);
  // CHECK-ASM: vsel
  vsc = vec_sel(vsc, vsc, vbc);
  // CHECK-ASM: vsel
  vuc = vec_sel(vuc, vuc, vuc);
  // CHECK-ASM: vsel
  vuc = vec_sel(vuc, vuc, vbc);
  // CHECK-ASM: vsel
  vbc = vec_sel(vbc, vbc, vuc);
  // CHECK-ASM: vsel
  vbc = vec_sel(vbc, vbc, vbc);
  // CHECK-ASM: vsel
  vss = vec_sel(vss, vss, vus);
  // CHECK-ASM: vsel
  vss = vec_sel(vss, vss, vbs);
  // CHECK-ASM: vsel
  vus = vec_sel(vus, vus, vus);
  // CHECK-ASM: vsel
  vus = vec_sel(vus, vus, vbs);
  // CHECK-ASM: vsel
  vbs = vec_sel(vbs, vbs, vus);
  // CHECK-ASM: vsel
  vbs = vec_sel(vbs, vbs, vbs);
  // CHECK-ASM: vsel
  vsi = vec_sel(vsi, vsi, vui);
  // CHECK-ASM: vsel
  vsi = vec_sel(vsi, vsi, vbi);
  // CHECK-ASM: vsel
  vui = vec_sel(vui, vui, vui);
  // CHECK-ASM: vsel
  vui = vec_sel(vui, vui, vbi);
  // CHECK-ASM: vsel
  vbi = vec_sel(vbi, vbi, vui);
  // CHECK-ASM: vsel
  vbi = vec_sel(vbi, vbi, vbi);
  // CHECK-ASM: vsel
  vsl = vec_sel(vsl, vsl, vul);
  // CHECK-ASM: vsel
  vsl = vec_sel(vsl, vsl, vbl);
  // CHECK-ASM: vsel
  vul = vec_sel(vul, vul, vul);
  // CHECK-ASM: vsel
  vul = vec_sel(vul, vul, vbl);
  // CHECK-ASM: vsel
  vbl = vec_sel(vbl, vbl, vul);
  // CHECK-ASM: vsel
  vbl = vec_sel(vbl, vbl, vbl);
  // CHECK-ASM: vsel
  vd = vec_sel(vd, vd, vul);
  // CHECK-ASM: vsel
  vd = vec_sel(vd, vd, vbl);
  // CHECK-ASM: vsel

  vsi = vec_gather_element(vsi, vui, cptrsi, 0);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vsi = vec_gather_element(vsi, vui, cptrsi, 1);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vsi = vec_gather_element(vsi, vui, cptrsi, 2);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vsi = vec_gather_element(vsi, vui, cptrsi, 3);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vui = vec_gather_element(vui, vui, cptrui, 0);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vui = vec_gather_element(vui, vui, cptrui, 1);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vui = vec_gather_element(vui, vui, cptrui, 2);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vui = vec_gather_element(vui, vui, cptrui, 3);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vbi = vec_gather_element(vbi, vui, cptrui, 0);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vbi = vec_gather_element(vbi, vui, cptrui, 1);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vbi = vec_gather_element(vbi, vui, cptrui, 2);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vbi = vec_gather_element(vbi, vui, cptrui, 3);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vsl = vec_gather_element(vsl, vul, cptrsl, 0);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vsl = vec_gather_element(vsl, vul, cptrsl, 1);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vul = vec_gather_element(vul, vul, cptrul, 0);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vul = vec_gather_element(vul, vul, cptrul, 1);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vbl = vec_gather_element(vbl, vul, cptrul, 0);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vbl = vec_gather_element(vbl, vul, cptrul, 1);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vd = vec_gather_element(vd, vul, cptrd, 0);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vd = vec_gather_element(vd, vul, cptrd, 1);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 1

  vec_scatter_element(vsi, vui, ptrsi, 0);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vsi, vui, ptrsi, 1);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vec_scatter_element(vsi, vui, ptrsi, 2);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vec_scatter_element(vsi, vui, ptrsi, 3);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vec_scatter_element(vui, vui, ptrui, 0);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vui, vui, ptrui, 1);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vec_scatter_element(vui, vui, ptrui, 2);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vec_scatter_element(vui, vui, ptrui, 3);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vec_scatter_element(vbi, vui, ptrui, 0);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vbi, vui, ptrui, 1);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vec_scatter_element(vbi, vui, ptrui, 2);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vec_scatter_element(vbi, vui, ptrui, 3);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vec_scatter_element(vsl, vul, ptrsl, 0);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vsl, vul, ptrsl, 1);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vec_scatter_element(vul, vul, ptrul, 0);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vul, vul, ptrul, 1);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vec_scatter_element(vbl, vul, ptrul, 0);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vbl, vul, ptrul, 1);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vec_scatter_element(vd, vul, ptrd, 0);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vd, vul, ptrd, 1);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 1

  vsc = vec_xl(idx, cptrsc);
  // CHECK-ASM: vl
  vuc = vec_xl(idx, cptruc);
  // CHECK-ASM: vl
  vss = vec_xl(idx, cptrss);
  // CHECK-ASM: vl
  vus = vec_xl(idx, cptrus);
  // CHECK-ASM: vl
  vsi = vec_xl(idx, cptrsi);
  // CHECK-ASM: vl
  vui = vec_xl(idx, cptrui);
  // CHECK-ASM: vl
  vsl = vec_xl(idx, cptrsl);
  // CHECK-ASM: vl
  vul = vec_xl(idx, cptrul);
  // CHECK-ASM: vl
  vd = vec_xl(idx, cptrd);
  // CHECK-ASM: vl

  vsc = vec_xld2(idx, cptrsc);
  // CHECK-ASM: vl
  vuc = vec_xld2(idx, cptruc);
  // CHECK-ASM: vl
  vss = vec_xld2(idx, cptrss);
  // CHECK-ASM: vl
  vus = vec_xld2(idx, cptrus);
  // CHECK-ASM: vl
  vsi = vec_xld2(idx, cptrsi);
  // CHECK-ASM: vl
  vui = vec_xld2(idx, cptrui);
  // CHECK-ASM: vl
  vsl = vec_xld2(idx, cptrsl);
  // CHECK-ASM: vl
  vul = vec_xld2(idx, cptrul);
  // CHECK-ASM: vl
  vd = vec_xld2(idx, cptrd);
  // CHECK-ASM: vl

  vsc = vec_xlw4(idx, cptrsc);
  // CHECK-ASM: vl
  vuc = vec_xlw4(idx, cptruc);
  // CHECK-ASM: vl
  vss = vec_xlw4(idx, cptrss);
  // CHECK-ASM: vl
  vus = vec_xlw4(idx, cptrus);
  // CHECK-ASM: vl
  vsi = vec_xlw4(idx, cptrsi);
  // CHECK-ASM: vl
  vui = vec_xlw4(idx, cptrui);
  // CHECK-ASM: vl

  vec_xst(vsc, idx, ptrsc);
  // CHECK-ASM: vst
  vec_xst(vuc, idx, ptruc);
  // CHECK-ASM: vst
  vec_xst(vss, idx, ptrss);
  // CHECK-ASM: vst
  vec_xst(vus, idx, ptrus);
  // CHECK-ASM: vst
  vec_xst(vsi, idx, ptrsi);
  // CHECK-ASM: vst
  vec_xst(vui, idx, ptrui);
  // CHECK-ASM: vst
  vec_xst(vsl, idx, ptrsl);
  // CHECK-ASM: vst
  vec_xst(vul, idx, ptrul);
  // CHECK-ASM: vst
  vec_xst(vd, idx, ptrd);
  // CHECK-ASM: vst

  vec_xstd2(vsc, idx, ptrsc);
  // CHECK-ASM: vst
  vec_xstd2(vuc, idx, ptruc);
  // CHECK-ASM: vst
  vec_xstd2(vss, idx, ptrss);
  // CHECK-ASM: vst
  vec_xstd2(vus, idx, ptrus);
  // CHECK-ASM: vst
  vec_xstd2(vsi, idx, ptrsi);
  // CHECK-ASM: vst
  vec_xstd2(vui, idx, ptrui);
  // CHECK-ASM: vst
  vec_xstd2(vsl, idx, ptrsl);
  // CHECK-ASM: vst
  vec_xstd2(vul, idx, ptrul);
  // CHECK-ASM: vst
  vec_xstd2(vd, idx, ptrd);
  // CHECK-ASM: vst

  vec_xstw4(vsc, idx, ptrsc);
  // CHECK-ASM: vst
  vec_xstw4(vuc, idx, ptruc);
  // CHECK-ASM: vst
  vec_xstw4(vss, idx, ptrss);
  // CHECK-ASM: vst
  vec_xstw4(vus, idx, ptrus);
  // CHECK-ASM: vst
  vec_xstw4(vsi, idx, ptrsi);
  // CHECK-ASM: vst
  vec_xstw4(vui, idx, ptrui);
  // CHECK-ASM: vst

  vsc = vec_load_bndry(cptrsc, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vuc = vec_load_bndry(cptruc, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vss = vec_load_bndry(cptrss, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vus = vec_load_bndry(cptrus, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vsi = vec_load_bndry(cptrsi, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vui = vec_load_bndry(cptrui, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vsl = vec_load_bndry(cptrsl, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vul = vec_load_bndry(cptrul, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vd = vec_load_bndry(cptrd, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vsc = vec_load_bndry(cptrsc, 128);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 1)
  // CHECK-ASM: vlbb
  vsc = vec_load_bndry(cptrsc, 256);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 2)
  // CHECK-ASM: vlbb
  vsc = vec_load_bndry(cptrsc, 512);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 3)
  // CHECK-ASM: vlbb
  vsc = vec_load_bndry(cptrsc, 1024);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 4)
  // CHECK-ASM: vlbb
  vsc = vec_load_bndry(cptrsc, 2048);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 5)
  // CHECK-ASM: vlbb
  vsc = vec_load_bndry(cptrsc, 4096);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(ptr %{{.*}}, i32 6)
  // CHECK-ASM: vlbb

  vsc = vec_load_len(cptrsc, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vuc = vec_load_len(cptruc, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vss = vec_load_len(cptrss, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vus = vec_load_len(cptrus, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vsi = vec_load_len(cptrsi, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vui = vec_load_len(cptrui, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vsl = vec_load_len(cptrsl, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vul = vec_load_len(cptrul, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll
  vd = vec_load_len(cptrd, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vll

  vec_store_len(vsc, ptrsc, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vuc, ptruc, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vss, ptrss, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vus, ptrus, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vsi, ptrsi, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vui, ptrui, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vsl, ptrsl, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vul, ptrul, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vd, ptrd, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, ptr %{{.*}})
  // CHECK-ASM: vstl

  vsl = vec_load_pair(sl + 1, sl - 1);
  // CHECK-ASM: vlvgp
  vul = vec_load_pair(ul + 1, ul - 1);
  // CHECK-ASM: vlvgp

  vuc = vec_genmask(0);
  // CHECK: <16 x i8> zeroinitializer
  vuc = vec_genmask(0x8000);
  // CHECK: <16 x i8> <i8 -1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  vuc = vec_genmask(0xffff);
  // CHECK: <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>

  vuc = vec_genmasks_8(0, 7);
  // CHECK: <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  vuc = vec_genmasks_8(1, 4);
  // CHECK: <16 x i8> <i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120, i8 120>
  vuc = vec_genmasks_8(6, 2);
  // CHECK: <16 x i8> <i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29, i8 -29>
  vus = vec_genmasks_16(0, 15);
  // CHECK: <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  vus = vec_genmasks_16(2, 11);
  // CHECK: <8 x i16> <i16 16368, i16 16368, i16 16368, i16 16368, i16 16368, i16 16368, i16 16368, i16 16368>
  vus = vec_genmasks_16(9, 2);
  // CHECK:  <8 x i16> <i16 -8065, i16 -8065, i16 -8065, i16 -8065, i16 -8065, i16 -8065, i16 -8065, i16 -8065>
  vui = vec_genmasks_32(0, 31);
  // CHECK: <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  vui = vec_genmasks_32(7, 20);
  // CHECK: <4 x i32> <i32 33552384, i32 33552384, i32 33552384, i32 33552384>
  vui = vec_genmasks_32(25, 4);
  // CHECK: <4 x i32> <i32 -134217601, i32 -134217601, i32 -134217601, i32 -134217601>
  vul = vec_genmasks_64(0, 63);
  // CHECK: <2 x i64> <i64 -1, i64 -1>
  vul = vec_genmasks_64(3, 40);
  // CHECK: <2 x i64> <i64 2305843009205305344, i64 2305843009205305344>
  vul = vec_genmasks_64(30, 11);
  // CHECK: <2 x i64> <i64 -4503582447501313, i64 -4503582447501313>

  vsc = vec_splat(vsc, 0);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> zeroinitializer
  // CHECK-ASM: vrepb
  vsc = vec_splat(vsc, 15);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  // CHECK-ASM: vrepb
  vuc = vec_splat(vuc, 0);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> zeroinitializer
  // CHECK-ASM: vrepb
  vuc = vec_splat(vuc, 15);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  // CHECK-ASM: vrepb
  vbc = vec_splat(vbc, 0);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> zeroinitializer
  // CHECK-ASM: vrepb
  vbc = vec_splat(vbc, 15);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  // CHECK-ASM: vrepb
  vss = vec_splat(vss, 0);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> zeroinitializer
  // CHECK-ASM: vreph
  vss = vec_splat(vss, 7);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  // CHECK-ASM: vreph
  vus = vec_splat(vus, 0);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> zeroinitializer
  // CHECK-ASM: vreph
  vus = vec_splat(vus, 7);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  // CHECK-ASM: vreph
  vbs = vec_splat(vbs, 0);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> zeroinitializer
  // CHECK-ASM: vreph
  vbs = vec_splat(vbs, 7);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  // CHECK-ASM: vreph
  vsi = vec_splat(vsi, 0);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK-ASM: vrepf
  vsi = vec_splat(vsi, 3);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  // CHECK-ASM: vrepf
  vui = vec_splat(vui, 0);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK-ASM: vrepf
  vui = vec_splat(vui, 3);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  // CHECK-ASM: vrepf
  vbi = vec_splat(vbi, 0);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK-ASM: vrepf
  vbi = vec_splat(vbi, 3);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  // CHECK-ASM: vrepf
  vsl = vec_splat(vsl, 0);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vrepg
  vsl = vec_splat(vsl, 1);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> <i32 1, i32 1>
  // CHECK-ASM: vrepg
  vul = vec_splat(vul, 0);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vrepg
  vul = vec_splat(vul, 1);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> <i32 1, i32 1>
  // CHECK-ASM: vrepg
  vbl = vec_splat(vbl, 0);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vrepg
  vbl = vec_splat(vbl, 1);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> <i32 1, i32 1>
  // CHECK-ASM: vrepg
  vd = vec_splat(vd, 0);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vrepg
  vd = vec_splat(vd, 1);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> <i32 1, i32 1>
  // CHECK-ASM: vrepg

  vsc = vec_splat_s8(-128);
  // CHECK: <16 x i8> <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  vsc = vec_splat_s8(127);
  // CHECK: <16 x i8> <i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127>
  vuc = vec_splat_u8(1);
  // CHECK: <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  vuc = vec_splat_u8(254);
  // CHECK: <16 x i8> <i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2, i8 -2>
  vss = vec_splat_s16(-32768);
  // CHECK: <8 x i16> <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  vss = vec_splat_s16(32767);
  // CHECK: <8 x i16> <i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767>
  vus = vec_splat_u16(1);
  // CHECK: <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  vus = vec_splat_u16(65534);
  // CHECK: <8 x i16> <i16 -2, i16 -2, i16 -2, i16 -2, i16 -2, i16 -2, i16 -2, i16 -2>
  vsi = vec_splat_s32(-32768);
  // CHECK: <4 x i32> <i32 -32768, i32 -32768, i32 -32768, i32 -32768>
  vsi = vec_splat_s32(32767);
  // CHECK: <4 x i32> <i32 32767, i32 32767, i32 32767, i32 32767>
  vui = vec_splat_u32(-32768);
  // CHECK: <4 x i32> <i32 -32768, i32 -32768, i32 -32768, i32 -32768>
  vui = vec_splat_u32(32767);
  // CHECK: <4 x i32> <i32 32767, i32 32767, i32 32767, i32 32767>
  vsl = vec_splat_s64(-32768);
  // CHECK: <2 x i64> <i64 -32768, i64 -32768>
  vsl = vec_splat_s64(32767);
  // CHECK: <2 x i64> <i64 32767, i64 32767>
  vul = vec_splat_u64(-32768);
  // CHECK: <2 x i64> <i64 -32768, i64 -32768>
  vul = vec_splat_u64(32767);
  // CHECK: <2 x i64> <i64 32767, i64 32767>

  vsc = vec_splats(sc);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> zeroinitializer
  // CHECK-ASM: vlrepb
  vuc = vec_splats(uc);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> poison, <16 x i32> zeroinitializer
  // CHECK-ASM: vlrepb
  vss = vec_splats(ss);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> zeroinitializer
  // CHECK-ASM: vlreph
  vus = vec_splats(us);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> zeroinitializer
  // CHECK-ASM: vlreph
  vsi = vec_splats(si);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK-ASM: vlrepf
  vui = vec_splats(ui);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK-ASM: vlrepf
  vsl = vec_splats(sl);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vlrepg
  vul = vec_splats(ul);
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vlrepg
  vd = vec_splats(d);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vlrepg

  vsl = vec_extend_s64(vsc);
  // CHECK-ASM: vsegb
  vsl = vec_extend_s64(vss);
  // CHECK-ASM: vsegh
  vsl = vec_extend_s64(vsi);
  // CHECK-ASM: vsegf

  vsc = vec_mergeh(vsc, vsc);
  // shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK-ASM: vmrhb
  vuc = vec_mergeh(vuc, vuc);
  // shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK-ASM: vmrhb
  vbc = vec_mergeh(vbc, vbc);
  // shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK-ASM: vmrhb
  vss = vec_mergeh(vss, vss);
  // shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK-ASM: vmrhh
  vus = vec_mergeh(vus, vus);
  // shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK-ASM: vmrhh
  vbs = vec_mergeh(vbs, vbs);
  // shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK-ASM: vmrhh
  vsi = vec_mergeh(vsi, vsi);
  // shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK-ASM: vmrhf
  vui = vec_mergeh(vui, vui);
  // shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK-ASM: vmrhf
  vbi = vec_mergeh(vbi, vbi);
  // shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK-ASM: vmrhf
  vsl = vec_mergeh(vsl, vsl);
  // shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK-ASM: vmrhg
  vul = vec_mergeh(vul, vul);
  // shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK-ASM: vmrhg
  vbl = vec_mergeh(vbl, vbl);
  // shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK-ASM: vmrhg
  vd = vec_mergeh(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK-ASM: vmrhg

  vsc = vec_mergel(vsc, vsc);
  // shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK-ASM: vmrlb
  vuc = vec_mergel(vuc, vuc);
  // shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK-ASM: vmrlb
  vbc = vec_mergel(vbc, vbc);
  // shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK-ASM: vmrlb
  vss = vec_mergel(vss, vss);
  // shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK-ASM: vmrlh
  vus = vec_mergel(vus, vus);
  // shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK-ASM: vmrlh
  vbs = vec_mergel(vbs, vbs);
  // shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK-ASM: vmrlh
  vsi = vec_mergel(vsi, vsi);
  // shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <i32 2, i32 6, i32 3, i32 7>
  // CHECK-ASM: vmrlf
  vui = vec_mergel(vui, vui);
  // shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <i32 2, i32 6, i32 3, i32 7>
  // CHECK-ASM: vmrlf
  vbi = vec_mergel(vbi, vbi);
  // shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <i32 2, i32 6, i32 3, i32 7>
  // CHECK-ASM: vmrlf
  vsl = vec_mergel(vsl, vsl);
  // shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <i32 1, i32 3>
  // CHECK-ASM: vmrlg
  vul = vec_mergel(vul, vul);
  // shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <i32 1, i32 3>
  // CHECK-ASM: vmrlg
  vbl = vec_mergel(vbl, vbl);
  // shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <i32 1, i32 3>
  // CHECK-ASM: vmrlg
  vd = vec_mergel(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <i32 1, i32 3>
  // CHECK-ASM: vmrlg

  vsc = vec_pack(vss, vss);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  // CHECK-ASM: vpkh
  vuc = vec_pack(vus, vus);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  // CHECK-ASM: vpkh
  vbc = vec_pack(vbs, vbs);
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  // CHECK-ASM: vpkh
  vss = vec_pack(vsi, vsi);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  // CHECK-ASM: vpkf
  vus = vec_pack(vui, vui);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  // CHECK-ASM: vpkf
  vbs = vec_pack(vbi, vbi);
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  // CHECK-ASM: vpkf
  vsi = vec_pack(vsl, vsl);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  // CHECK-ASM: vpkg
  vui = vec_pack(vul, vul);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  // CHECK-ASM: vpkg
  vbi = vec_pack(vbl, vbl);
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  // CHECK-ASM: vpkg

  vsc = vec_packs(vss, vss);
  // CHECK: call <16 x i8> @llvm.s390.vpksh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vpksh
  vuc = vec_packs(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vpklsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vpklsh
  vss = vec_packs(vsi, vsi);
  // CHECK: call <8 x i16> @llvm.s390.vpksf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vpksf
  vus = vec_packs(vui, vui);
  // CHECK: call <8 x i16> @llvm.s390.vpklsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vpklsf
  vsi = vec_packs(vsl, vsl);
  // CHECK: call <4 x i32> @llvm.s390.vpksg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vpksg
  vui = vec_packs(vul, vul);
  // CHECK: call <4 x i32> @llvm.s390.vpklsg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vpklsg

  vsc = vec_packs_cc(vss, vss, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vpkshs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vpkshs
  vuc = vec_packs_cc(vus, vus, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vpklshs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vpklshs
  vss = vec_packs_cc(vsi, vsi, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vpksfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vpksfs
  vus = vec_packs_cc(vui, vui, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vpklsfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vpklsfs
  vsi = vec_packs_cc(vsl, vsl, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vpksgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vpksgs
  vui = vec_packs_cc(vul, vul, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vpklsgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vpklsgs

  vuc = vec_packsu(vss, vss);
  // CHECK: call <16 x i8> @llvm.s390.vpklsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vpklsh
  vuc = vec_packsu(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vpklsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vpklsh
  vus = vec_packsu(vsi, vsi);
  // CHECK: call <8 x i16> @llvm.s390.vpklsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vpklsf
  vus = vec_packsu(vui, vui);
  // CHECK: call <8 x i16> @llvm.s390.vpklsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vpklsf
  vui = vec_packsu(vsl, vsl);
  // CHECK: call <4 x i32> @llvm.s390.vpklsg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vpklsg
  vui = vec_packsu(vul, vul);
  // CHECK: call <4 x i32> @llvm.s390.vpklsg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vpklsg

  vuc = vec_packsu_cc(vus, vus, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vpklshs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vpklshs
  vus = vec_packsu_cc(vui, vui, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vpklsfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vpklsfs
  vui = vec_packsu_cc(vul, vul, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vpklsgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vpklsgs

  vss = vec_unpackh(vsc);
  // CHECK: call <8 x i16> @llvm.s390.vuphb(<16 x i8> %{{.*}})
  // CHECK-ASM: vuphb
  vus = vec_unpackh(vuc);
  // CHECK: call <8 x i16> @llvm.s390.vuplhb(<16 x i8> %{{.*}})
  // CHECK-ASM: vuplhb
  vbs = vec_unpackh(vbc);
  // CHECK: call <8 x i16> @llvm.s390.vuphb(<16 x i8> %{{.*}})
  // CHECK-ASM: vuphb
  vsi = vec_unpackh(vss);
  // CHECK: call <4 x i32> @llvm.s390.vuphh(<8 x i16> %{{.*}})
  // CHECK-ASM: vuphh
  vui = vec_unpackh(vus);
  // CHECK: call <4 x i32> @llvm.s390.vuplhh(<8 x i16> %{{.*}})
  // CHECK-ASM: vuplhh
  vbi = vec_unpackh(vbs);
  // CHECK: call <4 x i32> @llvm.s390.vuphh(<8 x i16> %{{.*}})
  // CHECK-ASM: vuphh
  vsl = vec_unpackh(vsi);
  // CHECK: call <2 x i64> @llvm.s390.vuphf(<4 x i32> %{{.*}})
  // CHECK-ASM: vuphf
  vul = vec_unpackh(vui);
  // CHECK: call <2 x i64> @llvm.s390.vuplhf(<4 x i32> %{{.*}})
  // CHECK-ASM: vuplhf
  vbl = vec_unpackh(vbi);
  // CHECK: call <2 x i64> @llvm.s390.vuphf(<4 x i32> %{{.*}})
  // CHECK-ASM: vuphf

  vss = vec_unpackl(vsc);
  // CHECK: call <8 x i16> @llvm.s390.vuplb(<16 x i8> %{{.*}})
  // CHECK-ASM: vuplb
  vus = vec_unpackl(vuc);
  // CHECK: call <8 x i16> @llvm.s390.vupllb(<16 x i8> %{{.*}})
  // CHECK-ASM: vupllb
  vbs = vec_unpackl(vbc);
  // CHECK: call <8 x i16> @llvm.s390.vuplb(<16 x i8> %{{.*}})
  // CHECK-ASM: vuplb
  vsi = vec_unpackl(vss);
  // CHECK: call <4 x i32> @llvm.s390.vuplhw(<8 x i16> %{{.*}})
  // CHECK-ASM: vuplhw
  vui = vec_unpackl(vus);
  // CHECK: call <4 x i32> @llvm.s390.vupllh(<8 x i16> %{{.*}})
  // CHECK-ASM: vupllh
  vbi = vec_unpackl(vbs);
  // CHECK: call <4 x i32> @llvm.s390.vuplhw(<8 x i16> %{{.*}})
  // CHECK-ASM: vuplhw
  vsl = vec_unpackl(vsi);
  // CHECK: call <2 x i64> @llvm.s390.vuplf(<4 x i32> %{{.*}})
  // CHECK-ASM: vuplf
  vul = vec_unpackl(vui);
  // CHECK: call <2 x i64> @llvm.s390.vupllf(<4 x i32> %{{.*}})
  // CHECK-ASM: vupllf
  vbl = vec_unpackl(vbi);
  // CHECK: call <2 x i64> @llvm.s390.vuplf(<4 x i32> %{{.*}})
  // CHECK-ASM: vuplf
}

void test_compare(void) {
  // CHECK-ASM-LABEL: test_compare

  vbc = vec_cmpeq(vsc, vsc);
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqb
  vbc = vec_cmpeq(vuc, vuc);
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqb
  vbc = vec_cmpeq(vbc, vbc);
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqb
  vbs = vec_cmpeq(vss, vss);
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqh
  vbs = vec_cmpeq(vus, vus);
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqh
  vbs = vec_cmpeq(vbs, vbs);
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqh
  vbi = vec_cmpeq(vsi, vsi);
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqf
  vbi = vec_cmpeq(vui, vui);
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqf
  vbi = vec_cmpeq(vbi, vbi);
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqf
  vbl = vec_cmpeq(vsl, vsl);
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqg
  vbl = vec_cmpeq(vul, vul);
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqg
  vbl = vec_cmpeq(vbl, vbl);
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqg
  vbl = vec_cmpeq(vd, vd);
  // CHECK: fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfcedb

  vbc = vec_cmpge(vsc, vsc);
  // CHECK: icmp sge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchb
  vbc = vec_cmpge(vuc, vuc);
  // CHECK: icmp uge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlb
  vbs = vec_cmpge(vss, vss);
  // CHECK: icmp sge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchh
  vbs = vec_cmpge(vus, vus);
  // CHECK: icmp uge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlh
  vbi = vec_cmpge(vsi, vsi);
  // CHECK: icmp sge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchf
  vbi = vec_cmpge(vui, vui);
  // CHECK: icmp uge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlf
  vbl = vec_cmpge(vsl, vsl);
  // CHECK: icmp sge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchg
  vbl = vec_cmpge(vul, vul);
  // CHECK: icmp uge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlg
  vbl = vec_cmpge(vd, vd);
  // CHECK: fcmp oge <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchedb

  vbc = vec_cmpgt(vsc, vsc);
  // CHECK: icmp sgt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchb
  vbc = vec_cmpgt(vuc, vuc);
  // CHECK: icmp ugt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlb
  vbs = vec_cmpgt(vss, vss);
  // CHECK: icmp sgt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchh
  vbs = vec_cmpgt(vus, vus);
  // CHECK: icmp ugt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlh
  vbi = vec_cmpgt(vsi, vsi);
  // CHECK: icmp sgt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchf
  vbi = vec_cmpgt(vui, vui);
  // CHECK: icmp ugt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlf
  vbl = vec_cmpgt(vsl, vsl);
  // CHECK: icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchg
  vbl = vec_cmpgt(vul, vul);
  // CHECK: icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlg
  vbl = vec_cmpgt(vd, vd);
  // CHECK: fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchdb

  vbc = vec_cmple(vsc, vsc);
  // CHECK: icmp sle <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchb
  vbc = vec_cmple(vuc, vuc);
  // CHECK: icmp ule <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlb
  vbs = vec_cmple(vss, vss);
  // CHECK: icmp sle <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchh
  vbs = vec_cmple(vus, vus);
  // CHECK: icmp ule <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlh
  vbi = vec_cmple(vsi, vsi);
  // CHECK: icmp sle <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchf
  vbi = vec_cmple(vui, vui);
  // CHECK: icmp ule <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlf
  vbl = vec_cmple(vsl, vsl);
  // CHECK: icmp sle <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchg
  vbl = vec_cmple(vul, vul);
  // CHECK: icmp ule <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlg
  vbl = vec_cmple(vd, vd);
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchedb

  vbc = vec_cmplt(vsc, vsc);
  // CHECK: icmp slt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchb
  vbc = vec_cmplt(vuc, vuc);
  // CHECK: icmp ult <16 x i8> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlb
  vbs = vec_cmplt(vss, vss);
  // CHECK: icmp slt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchh
  vbs = vec_cmplt(vus, vus);
  // CHECK: icmp ult <8 x i16> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlh
  vbi = vec_cmplt(vsi, vsi);
  // CHECK: icmp slt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchf
  vbi = vec_cmplt(vui, vui);
  // CHECK: icmp ult <4 x i32> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlf
  vbl = vec_cmplt(vsl, vsl);
  // CHECK: icmp slt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchg
  vbl = vec_cmplt(vul, vul);
  // CHECK: icmp ult <2 x i64> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlg
  vbl = vec_cmplt(vd, vd);
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchdb

  idx = vec_all_eq(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_eq(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_eq(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_eq(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_eq(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_eq(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_eq(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_eq(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_eq(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_eq(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_eq(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_eq(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_eq(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_eq(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_eq(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_eq(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_eq(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_eq(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_eq(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_eq(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_eq(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_eq(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_eq(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_eq(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_eq(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_eq(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_eq(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_eq(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_eq(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_all_ne(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_ne(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_ne(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_ne(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_ne(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_ne(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_ne(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_all_ne(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_ne(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_ne(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_ne(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_ne(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_ne(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_ne(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_all_ne(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_ne(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_ne(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_ne(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_ne(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_ne(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_ne(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_all_ne(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_ne(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_ne(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_ne(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_ne(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_ne(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_ne(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_all_ne(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_all_ge(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_ge(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_ge(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_ge(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_ge(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_ge(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_ge(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_ge(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_ge(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_ge(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_ge(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_ge(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_ge(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_ge(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_ge(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_ge(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_ge(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_ge(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_ge(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_ge(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_ge(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_ge(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_ge(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_ge(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_ge(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_ge(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_ge(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_ge(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_ge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_all_gt(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_gt(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_gt(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_gt(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_gt(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_gt(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_gt(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_gt(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_gt(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_gt(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_gt(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_gt(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_gt(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_gt(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_gt(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_gt(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_gt(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_gt(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_gt(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_gt(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_gt(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_gt(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_gt(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_gt(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_gt(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_gt(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_gt(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_gt(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_gt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_all_le(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_le(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_le(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_le(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_le(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_le(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_le(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_le(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_le(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_le(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_le(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_le(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_le(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_le(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_le(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_le(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_le(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_le(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_le(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_le(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_le(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_le(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_le(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_le(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_le(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_le(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_le(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_le(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_le(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_all_lt(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_lt(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_lt(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_all_lt(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_lt(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_lt(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_lt(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_all_lt(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_lt(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_lt(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_all_lt(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_lt(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_lt(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_lt(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_all_lt(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_lt(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_lt(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_all_lt(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_lt(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_lt(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_lt(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_all_lt(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_lt(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_lt(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_all_lt(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_lt(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_lt(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_lt(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_all_lt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_all_nge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs
  idx = vec_all_ngt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs
  idx = vec_all_nle(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs
  idx = vec_all_nlt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_all_nan(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb
  idx = vec_all_numeric(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb

  idx = vec_any_eq(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_eq(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_eq(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_eq(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_eq(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_eq(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_eq(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_eq(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_eq(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_eq(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_eq(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_eq(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_eq(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_eq(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_eq(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_eq(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_eq(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_eq(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_eq(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_eq(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_eq(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_eq(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_eq(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_eq(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_eq(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_eq(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_eq(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_eq(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_eq(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_any_ne(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_ne(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_ne(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_ne(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_ne(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_ne(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_ne(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vceqbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vceqbs
  idx = vec_any_ne(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_ne(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_ne(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_ne(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_ne(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_ne(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_ne(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vceqhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vceqhs
  idx = vec_any_ne(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_ne(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_ne(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_ne(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_ne(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_ne(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_ne(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vceqfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vceqfs
  idx = vec_any_ne(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_ne(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_ne(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_ne(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_ne(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_ne(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_ne(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vceqgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vceqgs
  idx = vec_any_ne(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_any_ge(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_ge(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_ge(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_ge(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_ge(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_ge(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_ge(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_ge(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_ge(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_ge(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_ge(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_ge(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_ge(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_ge(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_ge(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_ge(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_ge(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_ge(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_ge(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_ge(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_ge(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_ge(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_ge(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_ge(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_ge(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_ge(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_ge(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_ge(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_ge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_any_gt(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_gt(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_gt(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_gt(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_gt(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_gt(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_gt(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_gt(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_gt(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_gt(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_gt(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_gt(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_gt(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_gt(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_gt(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_gt(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_gt(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_gt(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_gt(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_gt(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_gt(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_gt(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_gt(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_gt(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_gt(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_gt(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_gt(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_gt(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_gt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_le(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_le(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_le(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_le(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_le(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_le(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_le(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_le(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_le(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_le(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_le(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_le(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_le(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_le(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_le(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_le(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_le(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_le(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_le(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_le(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_le(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_le(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_le(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_le(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_le(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_le(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_le(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_le(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_le(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_any_lt(vsc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_lt(vsc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_lt(vbc, vsc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchbs
  idx = vec_any_lt(vuc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_lt(vuc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_lt(vbc, vuc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_lt(vbc, vbc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vchlbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vchlbs
  idx = vec_any_lt(vss, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_lt(vss, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_lt(vbs, vss);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchhs
  idx = vec_any_lt(vus, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_lt(vus, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_lt(vbs, vus);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_lt(vbs, vbs);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vchlhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vchlhs
  idx = vec_any_lt(vsi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_lt(vsi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_lt(vbi, vsi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchfs
  idx = vec_any_lt(vui, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_lt(vui, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_lt(vbi, vui);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_lt(vbi, vbi);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vchlfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vchlfs
  idx = vec_any_lt(vsl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_lt(vsl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_lt(vbl, vsl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchgs
  idx = vec_any_lt(vul, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_lt(vul, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_lt(vbl, vul);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_lt(vbl, vbl);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vchlgs(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vchlgs
  idx = vec_any_lt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_nge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs
  idx = vec_any_ngt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs
  idx = vec_any_nle(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs
  idx = vec_any_nlt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_nan(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb
  idx = vec_any_numeric(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb
}

void test_integer(void) {
  // CHECK-ASM-LABEL: test_integer

  vsc = vec_andc(vsc, vsc);
  // CHECK-ASM: vnc
  vsc = vec_andc(vsc, vbc);
  // CHECK-ASM: vnc
  vsc = vec_andc(vbc, vsc);
  // CHECK-ASM: vnc
  vuc = vec_andc(vuc, vuc);
  // CHECK-ASM: vnc
  vuc = vec_andc(vuc, vbc);
  // CHECK-ASM: vnc
  vuc = vec_andc(vbc, vuc);
  // CHECK-ASM: vnc
  vbc = vec_andc(vbc, vbc);
  // CHECK-ASM: vnc
  vss = vec_andc(vss, vss);
  // CHECK-ASM: vnc
  vss = vec_andc(vss, vbs);
  // CHECK-ASM: vnc
  vss = vec_andc(vbs, vss);
  // CHECK-ASM: vnc
  vus = vec_andc(vus, vus);
  // CHECK-ASM: vnc
  vus = vec_andc(vus, vbs);
  // CHECK-ASM: vnc
  vus = vec_andc(vbs, vus);
  // CHECK-ASM: vnc
  vbs = vec_andc(vbs, vbs);
  // CHECK-ASM: vnc
  vsi = vec_andc(vsi, vsi);
  // CHECK-ASM: vnc
  vsi = vec_andc(vsi, vbi);
  // CHECK-ASM: vnc
  vsi = vec_andc(vbi, vsi);
  // CHECK-ASM: vnc
  vui = vec_andc(vui, vui);
  // CHECK-ASM: vnc
  vui = vec_andc(vui, vbi);
  // CHECK-ASM: vnc
  vui = vec_andc(vbi, vui);
  // CHECK-ASM: vnc
  vbi = vec_andc(vbi, vbi);
  // CHECK-ASM: vnc
  vsl = vec_andc(vsl, vsl);
  // CHECK-ASM: vnc
  vsl = vec_andc(vsl, vbl);
  // CHECK-ASM: vnc
  vsl = vec_andc(vbl, vsl);
  // CHECK-ASM: vnc
  vul = vec_andc(vul, vul);
  // CHECK-ASM: vnc
  vul = vec_andc(vul, vbl);
  // CHECK-ASM: vnc
  vul = vec_andc(vbl, vul);
  // CHECK-ASM: vnc
  vbl = vec_andc(vbl, vbl);
  // CHECK-ASM: vnc
  vd = vec_andc(vd, vd);
  // CHECK-ASM: vnc
  vd = vec_andc(vd, vbl);
  // CHECK-ASM: vnc
  vd = vec_andc(vbl, vd);
  // CHECK-ASM: vnc

  vsc = vec_nor(vsc, vsc);
  // CHECK-ASM: vno
  vsc = vec_nor(vsc, vbc);
  // CHECK-ASM: vno
  vsc = vec_nor(vbc, vsc);
  // CHECK-ASM: vno
  vuc = vec_nor(vuc, vuc);
  // CHECK-ASM: vno
  vuc = vec_nor(vuc, vbc);
  // CHECK-ASM: vno
  vuc = vec_nor(vbc, vuc);
  // CHECK-ASM: vno
  vbc = vec_nor(vbc, vbc);
  // CHECK-ASM: vno
  vss = vec_nor(vss, vss);
  // CHECK-ASM: vno
  vss = vec_nor(vss, vbs);
  // CHECK-ASM: vno
  vss = vec_nor(vbs, vss);
  // CHECK-ASM: vno
  vus = vec_nor(vus, vus);
  // CHECK-ASM: vno
  vus = vec_nor(vus, vbs);
  // CHECK-ASM: vno
  vus = vec_nor(vbs, vus);
  // CHECK-ASM: vno
  vbs = vec_nor(vbs, vbs);
  // CHECK-ASM: vno
  vsi = vec_nor(vsi, vsi);
  // CHECK-ASM: vno
  vsi = vec_nor(vsi, vbi);
  // CHECK-ASM: vno
  vsi = vec_nor(vbi, vsi);
  // CHECK-ASM: vno
  vui = vec_nor(vui, vui);
  // CHECK-ASM: vno
  vui = vec_nor(vui, vbi);
  // CHECK-ASM: vno
  vui = vec_nor(vbi, vui);
  // CHECK-ASM: vno
  vbi = vec_nor(vbi, vbi);
  // CHECK-ASM: vno
  vsl = vec_nor(vsl, vsl);
  // CHECK-ASM: vno
  vsl = vec_nor(vsl, vbl);
  // CHECK-ASM: vno
  vsl = vec_nor(vbl, vsl);
  // CHECK-ASM: vno
  vul = vec_nor(vul, vul);
  // CHECK-ASM: vno
  vul = vec_nor(vul, vbl);
  // CHECK-ASM: vno
  vul = vec_nor(vbl, vul);
  // CHECK-ASM: vno
  vbl = vec_nor(vbl, vbl);
  // CHECK-ASM: vno
  vd = vec_nor(vd, vd);
  // CHECK-ASM: vno
  vd = vec_nor(vd, vbl);
  // CHECK-ASM: vno
  vd = vec_nor(vbl, vd);
  // CHECK-ASM: vno

  vuc = vec_cntlz(vsc);
  // CHECK: call range(i8 0, 9) <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %{{.*}}, i1 false)
  // CHECK-ASM: vclzb
  vuc = vec_cntlz(vuc);
  // CHECK: call range(i8 0, 9) <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %{{.*}}, i1 false)
  // CHECK-ASM: vclzb
  vus = vec_cntlz(vss);
  // CHECK: call range(i16 0, 17) <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %{{.*}}, i1 false)
  // CHECK-ASM: vclzh
  vus = vec_cntlz(vus);
  // CHECK: call range(i16 0, 17) <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %{{.*}}, i1 false)
  // CHECK-ASM: vclzh
  vui = vec_cntlz(vsi);
  // CHECK: call range(i32 0, 33) <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK-ASM: vclzf
  vui = vec_cntlz(vui);
  // CHECK: call range(i32 0, 33) <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK-ASM: vclzf
  vul = vec_cntlz(vsl);
  // CHECK: call range(i64 0, 65) <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK-ASM: vclzg
  vul = vec_cntlz(vul);
  // CHECK: call range(i64 0, 65) <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK-ASM: vclzg

  vuc = vec_cnttz(vsc);
  // CHECK: call range(i8 0, 9) <16 x i8> @llvm.cttz.v16i8(<16 x i8> %{{.*}}, i1 false)
  // CHECK-ASM: vctzb
  vuc = vec_cnttz(vuc);
  // CHECK: call range(i8 0, 9) <16 x i8> @llvm.cttz.v16i8(<16 x i8> %{{.*}}, i1 false)
  // CHECK-ASM: vctzb
  vus = vec_cnttz(vss);
  // CHECK: call range(i16 0, 17) <8 x i16> @llvm.cttz.v8i16(<8 x i16> %{{.*}}, i1 false)
  // CHECK-ASM: vctzh
  vus = vec_cnttz(vus);
  // CHECK: call range(i16 0, 17) <8 x i16> @llvm.cttz.v8i16(<8 x i16> %{{.*}}, i1 false)
  // CHECK-ASM: vctzh
  vui = vec_cnttz(vsi);
  // CHECK: call range(i32 0, 33) <4 x i32> @llvm.cttz.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK-ASM: vctzf
  vui = vec_cnttz(vui);
  // CHECK: call range(i32 0, 33) <4 x i32> @llvm.cttz.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK-ASM: vctzf
  vul = vec_cnttz(vsl);
  // CHECK: call range(i64 0, 65) <2 x i64> @llvm.cttz.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK-ASM: vctzg
  vul = vec_cnttz(vul);
  // CHECK: call range(i64 0, 65) <2 x i64> @llvm.cttz.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK-ASM: vctzg

  vuc = vec_popcnt(vsc);
  // CHECK: call range(i8 0, 9) <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %{{.*}})
  // CHECK-ASM: vpopct
  vuc = vec_popcnt(vuc);
  // CHECK: call range(i8 0, 9) <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %{{.*}})
  // CHECK-ASM: vpopct
  vus = vec_popcnt(vss);
  // CHECK: call range(i16 0, 17) <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %{{.*}})
  // (emulated)
  vus = vec_popcnt(vus);
  // CHECK: call range(i16 0, 17) <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %{{.*}})
  // (emulated)
  vui = vec_popcnt(vsi);
  // CHECK: call range(i32 0, 33) <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %{{.*}})
  // (emulated)
  vui = vec_popcnt(vui);
  // CHECK: call range(i32 0, 33) <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %{{.*}})
  // (emulated)
  vul = vec_popcnt(vsl);
  // CHECK: call range(i64 0, 65) <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %{{.*}})
  // (emulated)
  vul = vec_popcnt(vul);
  // CHECK: call range(i64 0, 65) <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %{{.*}})
  // (emulated)

  vsc = vec_rl(vsc, vuc);
  // CHECK: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: verllvb
  vuc = vec_rl(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: verllvb
  vss = vec_rl(vss, vus);
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: verllvh
  vus = vec_rl(vus, vus);
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: verllvh
  vsi = vec_rl(vsi, vui);
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: verllvf
  vui = vec_rl(vui, vui);
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: verllvf
  vsl = vec_rl(vsl, vul);
  // CHECK: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: verllvg
  vul = vec_rl(vul, vul);
  // CHECK: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: verllvg

  vsc = vec_rli(vsc, ul);
  // CHECK: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: verllb
  vuc = vec_rli(vuc, ul);
  // CHECK: call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: verllb
  vss = vec_rli(vss, ul);
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: verllh
  vus = vec_rli(vus, ul);
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: verllh
  vsi = vec_rli(vsi, ul);
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: verllf
  vui = vec_rli(vui, ul);
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: verllf
  vsl = vec_rli(vsl, ul);
  // CHECK: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: verllg
  vul = vec_rli(vul, ul);
  // CHECK: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: verllg

  vsc = vec_rl_mask(vsc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.verimb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: verimb
  vsc = vec_rl_mask(vsc, vuc, 255);
  // CHECK: call <16 x i8> @llvm.s390.verimb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: verimb
  vuc = vec_rl_mask(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.verimb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: verimb
  vuc = vec_rl_mask(vuc, vuc, 255);
  // CHECK: call <16 x i8> @llvm.s390.verimb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: verimb
  vss = vec_rl_mask(vss, vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.verimh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: verimh
  vss = vec_rl_mask(vss, vus, 255);
  // CHECK: call <8 x i16> @llvm.s390.verimh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 255)
  // CHECK-ASM: verimh
  vus = vec_rl_mask(vus, vus, 0);
  // CHECK: call <8 x i16> @llvm.s390.verimh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: verimh
  vus = vec_rl_mask(vus, vus, 255);
  // CHECK: call <8 x i16> @llvm.s390.verimh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 255)
  // CHECK-ASM: verimh
  vsi = vec_rl_mask(vsi, vui, 0);
  // CHECK: call <4 x i32> @llvm.s390.verimf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: verimf
  vsi = vec_rl_mask(vsi, vui, 255);
  // CHECK: call <4 x i32> @llvm.s390.verimf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 255)
  // CHECK-ASM: verimf
  vui = vec_rl_mask(vui, vui, 0);
  // CHECK: call <4 x i32> @llvm.s390.verimf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: verimf
  vui = vec_rl_mask(vui, vui, 255);
  // CHECK: call <4 x i32> @llvm.s390.verimf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 255)
  // CHECK-ASM: verimf
  vsl = vec_rl_mask(vsl, vul, 0);
  // CHECK: call <2 x i64> @llvm.s390.verimg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  // CHECK-ASM: verimg
  vsl = vec_rl_mask(vsl, vul, 255);
  // CHECK: call <2 x i64> @llvm.s390.verimg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 255)
  // CHECK-ASM: verimg
  vul = vec_rl_mask(vul, vul, 0);
  // CHECK: call <2 x i64> @llvm.s390.verimg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 0)
  // CHECK-ASM: verimg
  vul = vec_rl_mask(vul, vul, 255);
  // CHECK: call <2 x i64> @llvm.s390.verimg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i32 255)
  // CHECK-ASM: verimg

  vsc = vec_sll(vsc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsc = vec_sll(vsc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsc = vec_sll(vsc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vuc = vec_sll(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vuc = vec_sll(vuc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vuc = vec_sll(vuc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbc = vec_sll(vbc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbc = vec_sll(vbc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbc = vec_sll(vbc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vss = vec_sll(vss, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vss = vec_sll(vss, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vss = vec_sll(vss, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vus = vec_sll(vus, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vus = vec_sll(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vus = vec_sll(vus, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbs = vec_sll(vbs, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbs = vec_sll(vbs, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbs = vec_sll(vbs, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsi = vec_sll(vsi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsi = vec_sll(vsi, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsi = vec_sll(vsi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vui = vec_sll(vui, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vui = vec_sll(vui, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vui = vec_sll(vui, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbi = vec_sll(vbi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbi = vec_sll(vbi, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbi = vec_sll(vbi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsl = vec_sll(vsl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsl = vec_sll(vsl, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vsl = vec_sll(vsl, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vul = vec_sll(vul, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vul = vec_sll(vul, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vul = vec_sll(vul, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbl = vec_sll(vbl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbl = vec_sll(vbl, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl
  vbl = vec_sll(vbl, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsl

  vsc = vec_slb(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vsc = vec_slb(vsc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vuc = vec_slb(vuc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vuc = vec_slb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vss = vec_slb(vss, vss);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vss = vec_slb(vss, vus);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vus = vec_slb(vus, vss);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vus = vec_slb(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vsi = vec_slb(vsi, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vsi = vec_slb(vsi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vui = vec_slb(vui, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vui = vec_slb(vui, vui);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vsl = vec_slb(vsl, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vsl = vec_slb(vsl, vul);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vul = vec_slb(vul, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vul = vec_slb(vul, vul);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vd = vec_slb(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vd = vec_slb(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb

  vsc = vec_sld(vsc, vsc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vsc = vec_sld(vsc, vsc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vuc = vec_sld(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vuc = vec_sld(vuc, vuc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vbc = vec_sld(vbc, vbc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vbc = vec_sld(vbc, vbc, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vss = vec_sld(vss, vss, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vss = vec_sld(vss, vss, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vus = vec_sld(vus, vus, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vus = vec_sld(vus, vus, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vbs = vec_sld(vbs, vbs, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vbs = vec_sld(vbs, vbs, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vsi = vec_sld(vsi, vsi, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vsi = vec_sld(vsi, vsi, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vui = vec_sld(vui, vui, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vui = vec_sld(vui, vui, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vbi = vec_sld(vbi, vbi, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vbi = vec_sld(vbi, vbi, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vsl = vec_sld(vsl, vsl, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vsl = vec_sld(vsl, vsl, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vul = vec_sld(vul, vul, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vul = vec_sld(vul, vul, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vbl = vec_sld(vbl, vbl, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vbl = vec_sld(vbl, vbl, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vd = vec_sld(vd, vd, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vd = vec_sld(vd, vd, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb

  vsc = vec_sldw(vsc, vsc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vsc = vec_sldw(vsc, vsc, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vuc = vec_sldw(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vuc = vec_sldw(vuc, vuc, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vss = vec_sldw(vss, vss, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vss = vec_sldw(vss, vss, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vus = vec_sldw(vus, vus, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vus = vec_sldw(vus, vus, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vsi = vec_sldw(vsi, vsi, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vsi = vec_sldw(vsi, vsi, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vui = vec_sldw(vui, vui, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vui = vec_sldw(vui, vui, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vsl = vec_sldw(vsl, vsl, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vsl = vec_sldw(vsl, vsl, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vul = vec_sldw(vul, vul, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vul = vec_sldw(vul, vul, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb
  vd = vec_sldw(vd, vd, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vd = vec_sldw(vd, vd, 3);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vsldb

  vsc = vec_sral(vsc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsc = vec_sral(vsc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsc = vec_sral(vsc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vuc = vec_sral(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vuc = vec_sral(vuc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vuc = vec_sral(vuc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbc = vec_sral(vbc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbc = vec_sral(vbc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbc = vec_sral(vbc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vss = vec_sral(vss, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vss = vec_sral(vss, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vss = vec_sral(vss, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vus = vec_sral(vus, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vus = vec_sral(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vus = vec_sral(vus, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbs = vec_sral(vbs, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbs = vec_sral(vbs, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbs = vec_sral(vbs, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsi = vec_sral(vsi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsi = vec_sral(vsi, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsi = vec_sral(vsi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vui = vec_sral(vui, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vui = vec_sral(vui, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vui = vec_sral(vui, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbi = vec_sral(vbi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbi = vec_sral(vbi, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbi = vec_sral(vbi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsl = vec_sral(vsl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsl = vec_sral(vsl, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vsl = vec_sral(vsl, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vul = vec_sral(vul, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vul = vec_sral(vul, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vul = vec_sral(vul, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbl = vec_sral(vbl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbl = vec_sral(vbl, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra
  vbl = vec_sral(vbl, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsra(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsra

  vsc = vec_srab(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vsc = vec_srab(vsc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vuc = vec_srab(vuc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vuc = vec_srab(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vss = vec_srab(vss, vss);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vss = vec_srab(vss, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vus = vec_srab(vus, vss);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vus = vec_srab(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vsi = vec_srab(vsi, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vsi = vec_srab(vsi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vui = vec_srab(vui, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vui = vec_srab(vui, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vsl = vec_srab(vsl, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vsl = vec_srab(vsl, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vul = vec_srab(vul, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vul = vec_srab(vul, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vd = vec_srab(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vd = vec_srab(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab

  vsc = vec_srl(vsc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsc = vec_srl(vsc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsc = vec_srl(vsc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vuc = vec_srl(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vuc = vec_srl(vuc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vuc = vec_srl(vuc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbc = vec_srl(vbc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbc = vec_srl(vbc, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbc = vec_srl(vbc, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vss = vec_srl(vss, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vss = vec_srl(vss, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vss = vec_srl(vss, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vus = vec_srl(vus, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vus = vec_srl(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vus = vec_srl(vus, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbs = vec_srl(vbs, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbs = vec_srl(vbs, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbs = vec_srl(vbs, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsi = vec_srl(vsi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsi = vec_srl(vsi, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsi = vec_srl(vsi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vui = vec_srl(vui, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vui = vec_srl(vui, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vui = vec_srl(vui, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbi = vec_srl(vbi, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbi = vec_srl(vbi, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbi = vec_srl(vbi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsl = vec_srl(vsl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsl = vec_srl(vsl, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vsl = vec_srl(vsl, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vul = vec_srl(vul, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vul = vec_srl(vul, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vul = vec_srl(vul, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbl = vec_srl(vbl, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbl = vec_srl(vbl, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl
  vbl = vec_srl(vbl, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrl(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrl

  vsc = vec_srb(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vsc = vec_srb(vsc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vuc = vec_srb(vuc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vuc = vec_srb(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vss = vec_srb(vss, vss);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vss = vec_srb(vss, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vus = vec_srb(vus, vss);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vus = vec_srb(vus, vus);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vsi = vec_srb(vsi, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vsi = vec_srb(vsi, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vui = vec_srb(vui, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vui = vec_srb(vui, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vsl = vec_srb(vsl, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vsl = vec_srb(vsl, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vul = vec_srb(vul, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vul = vec_srb(vul, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vd = vec_srb(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vd = vec_srb(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb

  vsc = vec_abs(vsc);
  // CHECK-ASM: vlcb
  vss = vec_abs(vss);
  // CHECK-ASM: vlch
  vsi = vec_abs(vsi);
  // CHECK-ASM: vlcf
  vsl = vec_abs(vsl);
  // CHECK-ASM: vlcg

  vsc = vec_max(vsc, vsc);
  // CHECK-ASM: vmxb
  vsc = vec_max(vsc, vbc);
  // CHECK-ASM: vmxb
  vsc = vec_max(vbc, vsc);
  // CHECK-ASM: vmxb
  vuc = vec_max(vuc, vuc);
  // CHECK-ASM: vmxlb
  vuc = vec_max(vuc, vbc);
  // CHECK-ASM: vmxlb
  vuc = vec_max(vbc, vuc);
  // CHECK-ASM: vmxlb
  vss = vec_max(vss, vss);
  // CHECK-ASM: vmxh
  vss = vec_max(vss, vbs);
  // CHECK-ASM: vmxh
  vss = vec_max(vbs, vss);
  // CHECK-ASM: vmxh
  vus = vec_max(vus, vus);
  // CHECK-ASM: vmxlh
  vus = vec_max(vus, vbs);
  // CHECK-ASM: vmxlh
  vus = vec_max(vbs, vus);
  // CHECK-ASM: vmxlh
  vsi = vec_max(vsi, vsi);
  // CHECK-ASM: vmxf
  vsi = vec_max(vsi, vbi);
  // CHECK-ASM: vmxf
  vsi = vec_max(vbi, vsi);
  // CHECK-ASM: vmxf
  vui = vec_max(vui, vui);
  // CHECK-ASM: vmxlf
  vui = vec_max(vui, vbi);
  // CHECK-ASM: vmxlf
  vui = vec_max(vbi, vui);
  // CHECK-ASM: vmxlf
  vsl = vec_max(vsl, vsl);
  // CHECK-ASM: vmxg
  vsl = vec_max(vsl, vbl);
  // CHECK-ASM: vmxg
  vsl = vec_max(vbl, vsl);
  // CHECK-ASM: vmxg
  vul = vec_max(vul, vul);
  // CHECK-ASM: vmxlg
  vul = vec_max(vul, vbl);
  // CHECK-ASM: vmxlg
  vul = vec_max(vbl, vul);
  // CHECK-ASM: vmxlg
  vd = vec_max(vd, vd);
  // (emulated)

  vsc = vec_min(vsc, vsc);
  // CHECK-ASM: vmnb
  vsc = vec_min(vsc, vbc);
  // CHECK-ASM: vmnb
  vsc = vec_min(vbc, vsc);
  // CHECK-ASM: vmnb
  vuc = vec_min(vuc, vuc);
  // CHECK-ASM: vmnlb
  vuc = vec_min(vuc, vbc);
  // CHECK-ASM: vmnlb
  vuc = vec_min(vbc, vuc);
  // CHECK-ASM: vmnlb
  vss = vec_min(vss, vss);
  // CHECK-ASM: vmnh
  vss = vec_min(vss, vbs);
  // CHECK-ASM: vmnh
  vss = vec_min(vbs, vss);
  // CHECK-ASM: vmnh
  vus = vec_min(vus, vus);
  // CHECK-ASM: vmnlh
  vus = vec_min(vus, vbs);
  // CHECK-ASM: vmnlh
  vus = vec_min(vbs, vus);
  // CHECK-ASM: vmnlh
  vsi = vec_min(vsi, vsi);
  // CHECK-ASM: vmnf
  vsi = vec_min(vsi, vbi);
  // CHECK-ASM: vmnf
  vsi = vec_min(vbi, vsi);
  // CHECK-ASM: vmnf
  vui = vec_min(vui, vui);
  // CHECK-ASM: vmnlf
  vui = vec_min(vui, vbi);
  // CHECK-ASM: vmnlf
  vui = vec_min(vbi, vui);
  // CHECK-ASM: vmnlf
  vsl = vec_min(vsl, vsl);
  // CHECK-ASM: vmng
  vsl = vec_min(vsl, vbl);
  // CHECK-ASM: vmng
  vsl = vec_min(vbl, vsl);
  // CHECK-ASM: vmng
  vul = vec_min(vul, vul);
  // CHECK-ASM: vmnlg
  vul = vec_min(vul, vbl);
  // CHECK-ASM: vmnlg
  vul = vec_min(vbl, vul);
  // CHECK-ASM: vmnlg
  vd = vec_min(vd, vd);
  // (emulated)

  vuc = vec_addc(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vaccb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vaccb
  vus = vec_addc(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vacch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vacch
  vui = vec_addc(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vaccf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vaccf
  vul = vec_addc(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vaccg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vaccg

  vuc = vec_add_u128(vuc, vuc);
  // CHECK-ASM: vaq
  vuc = vec_addc_u128(vuc, vuc);
  // CHECK: call i128 @llvm.s390.vaccq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vaccq
  vuc = vec_adde_u128(vuc, vuc, vuc);
  // CHECK: call i128 @llvm.s390.vacq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vacq
  vuc = vec_addec_u128(vuc, vuc, vuc);
  // CHECK: call i128 @llvm.s390.vacccq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vacccq

  vsc = vec_avg(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vavgb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vavgb
  vuc = vec_avg(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vavglb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vavglb
  vss = vec_avg(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vavgh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vavgh
  vus = vec_avg(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vavglh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vavglh
  vsi = vec_avg(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vavgf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vavgf
  vui = vec_avg(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vavglf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vavglf
  vsl = vec_avg(vsl, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vavgg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vavgg
  vul = vec_avg(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vavglg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vavglg

  vui = vec_checksum(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vcksm(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vcksm

  vus = vec_gfmsum(vuc, vuc);
  // CHECK: call <8 x i16> @llvm.s390.vgfmb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vgfmb
  vui = vec_gfmsum(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vgfmh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vgfmh
  vul = vec_gfmsum(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vgfmf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vgfmf
  vuc = vec_gfmsum_128(vul, vul);
  // CHECK: call i128 @llvm.s390.vgfmg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vgfmg

  vus = vec_gfmsum_accum(vuc, vuc, vus);
  // CHECK: call <8 x i16> @llvm.s390.vgfmab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vgfmab
  vui = vec_gfmsum_accum(vus, vus, vui);
  // CHECK: call <4 x i32> @llvm.s390.vgfmah(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vgfmah
  vul = vec_gfmsum_accum(vui, vui, vul);
  // CHECK: call <2 x i64> @llvm.s390.vgfmaf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vgfmaf
  vuc = vec_gfmsum_accum_128(vul, vul, vuc);
  // CHECK: call i128 @llvm.s390.vgfmag(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vgfmag

  vsc = vec_mladd(vsc, vsc, vsc);
  // CHECK-ASM: vmalb
  vsc = vec_mladd(vuc, vsc, vsc);
  // CHECK-ASM: vmalb
  vsc = vec_mladd(vsc, vuc, vuc);
  // CHECK-ASM: vmalb
  vuc = vec_mladd(vuc, vuc, vuc);
  // CHECK-ASM: vmalb
  vss = vec_mladd(vss, vss, vss);
  // CHECK-ASM: vmalhw
  vss = vec_mladd(vus, vss, vss);
  // CHECK-ASM: vmalhw
  vss = vec_mladd(vss, vus, vus);
  // CHECK-ASM: vmalhw
  vus = vec_mladd(vus, vus, vus);
  // CHECK-ASM: vmalhw
  vsi = vec_mladd(vsi, vsi, vsi);
  // CHECK-ASM: vmalf
  vsi = vec_mladd(vui, vsi, vsi);
  // CHECK-ASM: vmalf
  vsi = vec_mladd(vsi, vui, vui);
  // CHECK-ASM: vmalf
  vui = vec_mladd(vui, vui, vui);
  // CHECK-ASM: vmalf

  vsc = vec_mhadd(vsc, vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vmahb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmahb
  vuc = vec_mhadd(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vmalhb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmalhb
  vss = vec_mhadd(vss, vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmahh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmahh
  vus = vec_mhadd(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmalhh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmalhh
  vsi = vec_mhadd(vsi, vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmahf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmahf
  vui = vec_mhadd(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmalhf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmalhf

  vss = vec_meadd(vsc, vsc, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmaeb
  vus = vec_meadd(vuc, vuc, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmaleb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmaleb
  vsi = vec_meadd(vss, vss, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmaeh
  vui = vec_meadd(vus, vus, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmaleh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmaleh
  vsl = vec_meadd(vsi, vsi, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmaef
  vul = vec_meadd(vui, vui, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmalef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmalef

  vss = vec_moadd(vsc, vsc, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmaob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmaob
  vus = vec_moadd(vuc, vuc, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmalob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmalob
  vsi = vec_moadd(vss, vss, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmaoh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmaoh
  vui = vec_moadd(vus, vus, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmaloh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmaloh
  vsl = vec_moadd(vsi, vsi, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmaof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmaof
  vul = vec_moadd(vui, vui, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmalof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmalof

  vsc = vec_mulh(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vmhb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmhb
  vuc = vec_mulh(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vmlhb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmlhb
  vss = vec_mulh(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vmhh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmhh
  vus = vec_mulh(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vmlhh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmlhh
  vsi = vec_mulh(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vmhf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmhf
  vui = vec_mulh(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vmlhf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmlhf

  vss = vec_mule(vsc, vsc);
  // CHECK: call <8 x i16> @llvm.s390.vmeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmeb
  vus = vec_mule(vuc, vuc);
  // CHECK: call <8 x i16> @llvm.s390.vmleb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmleb
  vsi = vec_mule(vss, vss);
  // CHECK: call <4 x i32> @llvm.s390.vmeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmeh
  vui = vec_mule(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vmleh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmleh
  vsl = vec_mule(vsi, vsi);
  // CHECK: call <2 x i64> @llvm.s390.vmef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmef
  vul = vec_mule(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vmlef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmlef

  vss = vec_mulo(vsc, vsc);
  // CHECK: call <8 x i16> @llvm.s390.vmob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmob
  vus = vec_mulo(vuc, vuc);
  // CHECK: call <8 x i16> @llvm.s390.vmlob(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vmlob
  vsi = vec_mulo(vss, vss);
  // CHECK: call <4 x i32> @llvm.s390.vmoh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmoh
  vui = vec_mulo(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vmloh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vmloh
  vsl = vec_mulo(vsi, vsi);
  // CHECK: call <2 x i64> @llvm.s390.vmof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmof
  vul = vec_mulo(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vmlof(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vmlof

  vuc = vec_subc(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vscbib(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vscbib
  vus = vec_subc(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vscbih(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vscbih
  vui = vec_subc(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vscbif(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vscbif
  vul = vec_subc(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vscbig(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vscbig

  vuc = vec_sub_u128(vuc, vuc);
  // CHECK-ASM: vsq
  vuc = vec_subc_u128(vuc, vuc);
  // CHECK: call i128 @llvm.s390.vscbiq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vscbiq
  vuc = vec_sube_u128(vuc, vuc, vuc);
  // CHECK: call i128 @llvm.s390.vsbiq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vsbiq
  vuc = vec_subec_u128(vuc, vuc, vuc);
  // CHECK: call i128 @llvm.s390.vsbcbiq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vsbcbiq

  vui = vec_sum4(vuc, vuc);
  // CHECK: call <4 x i32> @llvm.s390.vsumb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsumb
  vui = vec_sum4(vus, vus);
  // CHECK: call <4 x i32> @llvm.s390.vsumh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vsumh
  vul = vec_sum2(vus, vus);
  // CHECK: call <2 x i64> @llvm.s390.vsumgh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vsumgh
  vul = vec_sum2(vui, vui);
  // CHECK: call <2 x i64> @llvm.s390.vsumgf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vsumgf
  vuc = vec_sum_u128(vui, vui);
  // CHECK: call i128 @llvm.s390.vsumqf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vsumqf
  vuc = vec_sum_u128(vul, vul);
  // CHECK: call i128 @llvm.s390.vsumqg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vsumqg

  idx = vec_test_mask(vsc, vuc);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vuc, vuc);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vss, vus);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vus, vus);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vsi, vui);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vui, vui);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vsl, vul);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vul, vul);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vd, vul);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
}

void test_string(void) {
  // CHECK-ASM-LABEL: test_string

  vsc = vec_cp_until_zero(vsc);
  // CHECK: call <16 x i8> @llvm.s390.vistrb(<16 x i8> %{{.*}})
  // CHECK-ASM: vistrb
  vuc = vec_cp_until_zero(vuc);
  // CHECK: call <16 x i8> @llvm.s390.vistrb(<16 x i8> %{{.*}})
  // CHECK-ASM: vistrb
  vbc = vec_cp_until_zero(vbc);
  // CHECK: call <16 x i8> @llvm.s390.vistrb(<16 x i8> %{{.*}})
  // CHECK-ASM: vistrb
  vss = vec_cp_until_zero(vss);
  // CHECK: call <8 x i16> @llvm.s390.vistrh(<8 x i16> %{{.*}})
  // CHECK-ASM: vistrh
  vus = vec_cp_until_zero(vus);
  // CHECK: call <8 x i16> @llvm.s390.vistrh(<8 x i16> %{{.*}})
  // CHECK-ASM: vistrh
  vbs = vec_cp_until_zero(vbs);
  // CHECK: call <8 x i16> @llvm.s390.vistrh(<8 x i16> %{{.*}})
  // CHECK-ASM: vistrh
  vsi = vec_cp_until_zero(vsi);
  // CHECK: call <4 x i32> @llvm.s390.vistrf(<4 x i32> %{{.*}})
  // CHECK-ASM: vistrf
  vui = vec_cp_until_zero(vui);
  // CHECK: call <4 x i32> @llvm.s390.vistrf(<4 x i32> %{{.*}})
  // CHECK-ASM: vistrf
  vbi = vec_cp_until_zero(vbi);
  // CHECK: call <4 x i32> @llvm.s390.vistrf(<4 x i32> %{{.*}})
  // CHECK-ASM: vistrf

  vsc = vec_cp_until_zero_cc(vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vistrbs(<16 x i8> %{{.*}})
  // CHECK-ASM: vistrbs
  vuc = vec_cp_until_zero_cc(vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vistrbs(<16 x i8> %{{.*}})
  // CHECK-ASM: vistrbs
  vbc = vec_cp_until_zero_cc(vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vistrbs(<16 x i8> %{{.*}})
  // CHECK-ASM: vistrbs
  vss = vec_cp_until_zero_cc(vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vistrhs(<8 x i16> %{{.*}})
  // CHECK-ASM: vistrhs
  vus = vec_cp_until_zero_cc(vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vistrhs(<8 x i16> %{{.*}})
  // CHECK-ASM: vistrhs
  vbs = vec_cp_until_zero_cc(vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vistrhs(<8 x i16> %{{.*}})
  // CHECK-ASM: vistrhs
  vsi = vec_cp_until_zero_cc(vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vistrfs(<4 x i32> %{{.*}})
  // CHECK-ASM: vistrfs
  vui = vec_cp_until_zero_cc(vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vistrfs(<4 x i32> %{{.*}})
  // CHECK-ASM: vistrfs
  vbi = vec_cp_until_zero_cc(vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vistrfs(<4 x i32> %{{.*}})
  // CHECK-ASM: vistrfs

  vsc = vec_cmpeq_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfeeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeeb
  vuc = vec_cmpeq_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfeeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeeb
  vuc = vec_cmpeq_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfeeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeeb
  vss = vec_cmpeq_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfeeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeeh
  vus = vec_cmpeq_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfeeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeeh
  vus = vec_cmpeq_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfeeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeeh
  vsi = vec_cmpeq_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfeef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeef
  vui = vec_cmpeq_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfeef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeef
  vui = vec_cmpeq_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfeef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeef

  vsc = vec_cmpeq_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeebs
  vuc = vec_cmpeq_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeebs
  vuc = vec_cmpeq_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeebs
  vss = vec_cmpeq_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeehs
  vus = vec_cmpeq_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeehs
  vus = vec_cmpeq_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeehs
  vsi = vec_cmpeq_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeefs
  vui = vec_cmpeq_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeefs
  vui = vec_cmpeq_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeefs

  vsc = vec_cmpeq_or_0_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfeezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeezb
  vuc = vec_cmpeq_or_0_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfeezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeezb
  vuc = vec_cmpeq_or_0_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfeezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeezb
  vss = vec_cmpeq_or_0_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfeezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeezh
  vus = vec_cmpeq_or_0_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfeezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeezh
  vus = vec_cmpeq_or_0_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfeezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeezh
  vsi = vec_cmpeq_or_0_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfeezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeezf
  vui = vec_cmpeq_or_0_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfeezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeezf
  vui = vec_cmpeq_or_0_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfeezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeezf

  vsc = vec_cmpeq_or_0_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeezbs
  vuc = vec_cmpeq_or_0_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeezbs
  vuc = vec_cmpeq_or_0_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfeezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeezbs
  vss = vec_cmpeq_or_0_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeezhs
  vus = vec_cmpeq_or_0_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeezhs
  vus = vec_cmpeq_or_0_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfeezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeezhs
  vsi = vec_cmpeq_or_0_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeezfs
  vui = vec_cmpeq_or_0_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeezfs
  vui = vec_cmpeq_or_0_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfeezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfeezfs

  vsc = vec_cmpne_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfeneb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeneb
  vuc = vec_cmpne_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfeneb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeneb
  vuc = vec_cmpne_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfeneb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfeneb
  vss = vec_cmpne_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfeneh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeneh
  vus = vec_cmpne_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfeneh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeneh
  vus = vec_cmpne_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfeneh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfeneh
  vsi = vec_cmpne_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfenef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenef
  vui = vec_cmpne_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfenef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenef
  vui = vec_cmpne_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfenef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenef

  vsc = vec_cmpne_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenebs
  vuc = vec_cmpne_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenebs
  vuc = vec_cmpne_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenebs
  vss = vec_cmpne_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenehs
  vus = vec_cmpne_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenehs
  vus = vec_cmpne_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenehs
  vsi = vec_cmpne_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenefs
  vui = vec_cmpne_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenefs
  vui = vec_cmpne_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenefs

  vsc = vec_cmpne_or_0_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfenezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenezb
  vuc = vec_cmpne_or_0_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfenezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenezb
  vuc = vec_cmpne_or_0_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfenezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenezb
  vss = vec_cmpne_or_0_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfenezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenezh
  vus = vec_cmpne_or_0_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfenezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenezh
  vus = vec_cmpne_or_0_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfenezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenezh
  vsi = vec_cmpne_or_0_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfenezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenezf
  vui = vec_cmpne_or_0_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfenezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenezf
  vui = vec_cmpne_or_0_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfenezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenezf

  vsc = vec_cmpne_or_0_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenezbs
  vuc = vec_cmpne_or_0_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenezbs
  vuc = vec_cmpne_or_0_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfenezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vfenezbs
  vss = vec_cmpne_or_0_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenezhs
  vus = vec_cmpne_or_0_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenezhs
  vus = vec_cmpne_or_0_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfenezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK-ASM: vfenezhs
  vsi = vec_cmpne_or_0_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenezfs
  vui = vec_cmpne_or_0_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenezfs
  vui = vec_cmpne_or_0_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfenezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK-ASM: vfenezfs

  vbc = vec_cmprg(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vstrcb
  vbs = vec_cmprg(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vstrch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vstrch
  vbi = vec_cmprg(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vstrcf

  vbc = vec_cmprg_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrcbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vstrcbs
  vbs = vec_cmprg_cc(vus, vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrchs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vstrchs
  vbi = vec_cmprg_cc(vui, vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrcfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vstrcfs

  vuc = vec_cmprg_idx(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vstrcb
  vus = vec_cmprg_idx(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vstrch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vstrch
  vui = vec_cmprg_idx(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vstrcf

  vuc = vec_cmprg_idx_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrcbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vstrcbs
  vus = vec_cmprg_idx_cc(vus, vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrchs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vstrchs
  vui = vec_cmprg_idx_cc(vui, vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrcfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vstrcfs

  vuc = vec_cmprg_or_0_idx(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vstrczb
  vus = vec_cmprg_or_0_idx(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vstrczh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vstrczh
  vui = vec_cmprg_or_0_idx(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vstrczf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vstrczf

  vuc = vec_cmprg_or_0_idx_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrczbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vstrczbs
  vus = vec_cmprg_or_0_idx_cc(vus, vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrczhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vstrczhs
  vui = vec_cmprg_or_0_idx_cc(vui, vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrczfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vstrczfs

  vbc = vec_cmpnrg(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vstrcb
  vbs = vec_cmpnrg(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vstrch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vstrch
  vbi = vec_cmpnrg(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vstrcf

  vbc = vec_cmpnrg_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrcbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vstrcbs
  vbs = vec_cmpnrg_cc(vus, vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrchs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vstrchs
  vbi = vec_cmpnrg_cc(vui, vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrcfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vstrcfs

  vuc = vec_cmpnrg_idx(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vstrcb
  vus = vec_cmpnrg_idx(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vstrch(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vstrch
  vui = vec_cmpnrg_idx(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vstrcf

  vuc = vec_cmpnrg_idx_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrcbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vstrcbs
  vus = vec_cmpnrg_idx_cc(vus, vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrchs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vstrchs
  vui = vec_cmpnrg_idx_cc(vui, vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrcfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vstrcfs

  vuc = vec_cmpnrg_or_0_idx(vuc, vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vstrczb
  vus = vec_cmpnrg_or_0_idx(vus, vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vstrczh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vstrczh
  vui = vec_cmpnrg_or_0_idx(vui, vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vstrczf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vstrczf

  vuc = vec_cmpnrg_or_0_idx_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrczbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vstrczbs
  vus = vec_cmpnrg_or_0_idx_cc(vus, vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vstrczhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vstrczhs
  vui = vec_cmpnrg_or_0_idx_cc(vui, vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vstrczfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vstrczfs

  vbc = vec_find_any_eq(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vfaeb
  vbc = vec_find_any_eq(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vfaeb
  vbc = vec_find_any_eq(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vfaeb
  vbs = vec_find_any_eq(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vfaeh
  vbs = vec_find_any_eq(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vfaeh
  vbs = vec_find_any_eq(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vfaeh
  vbi = vec_find_any_eq(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vfaef
  vbi = vec_find_any_eq(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vfaef
  vbi = vec_find_any_eq(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vfaef

  vbc = vec_find_any_eq_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vfaebs
  vbc = vec_find_any_eq_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vfaebs
  vbc = vec_find_any_eq_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vfaebs
  vbs = vec_find_any_eq_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vfaehs
  vbs = vec_find_any_eq_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vfaehs
  vbs = vec_find_any_eq_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 4)
  // CHECK-ASM: vfaehs
  vbi = vec_find_any_eq_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vfaefs
  vbi = vec_find_any_eq_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vfaefs
  vbi = vec_find_any_eq_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 4)
  // CHECK-ASM: vfaefs

  vsc = vec_find_any_eq_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaeb
  vuc = vec_find_any_eq_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaeb
  vuc = vec_find_any_eq_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaeb
  vss = vec_find_any_eq_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaeh
  vus = vec_find_any_eq_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaeh
  vus = vec_find_any_eq_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaeh
  vsi = vec_find_any_eq_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaef
  vui = vec_find_any_eq_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaef
  vui = vec_find_any_eq_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaef

  vsc = vec_find_any_eq_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaebs
  vuc = vec_find_any_eq_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaebs
  vuc = vec_find_any_eq_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaebs
  vss = vec_find_any_eq_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaehs
  vus = vec_find_any_eq_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaehs
  vus = vec_find_any_eq_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaehs
  vsi = vec_find_any_eq_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaefs
  vui = vec_find_any_eq_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaefs
  vui = vec_find_any_eq_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaefs

  vsc = vec_find_any_eq_or_0_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezb
  vuc = vec_find_any_eq_or_0_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezb
  vuc = vec_find_any_eq_or_0_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezb
  vss = vec_find_any_eq_or_0_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezh
  vus = vec_find_any_eq_or_0_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezh
  vus = vec_find_any_eq_or_0_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezh
  vsi = vec_find_any_eq_or_0_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezf
  vui = vec_find_any_eq_or_0_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezf
  vui = vec_find_any_eq_or_0_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezf

  vsc = vec_find_any_eq_or_0_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezbs
  vuc = vec_find_any_eq_or_0_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezbs
  vuc = vec_find_any_eq_or_0_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezbs
  vss = vec_find_any_eq_or_0_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezhs
  vus = vec_find_any_eq_or_0_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezhs
  vus = vec_find_any_eq_or_0_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezhs
  vsi = vec_find_any_eq_or_0_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezfs
  vui = vec_find_any_eq_or_0_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezfs
  vui = vec_find_any_eq_or_0_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  // CHECK-ASM: vfaezfs

  vbc = vec_find_any_ne(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vfaeb
  vbc = vec_find_any_ne(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vfaeb
  vbc = vec_find_any_ne(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vfaeb
  vbs = vec_find_any_ne(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vfaeh
  vbs = vec_find_any_ne(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vfaeh
  vbs = vec_find_any_ne(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vfaeh
  vbi = vec_find_any_ne(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vfaef
  vbi = vec_find_any_ne(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vfaef
  vbi = vec_find_any_ne(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vfaef

  vbc = vec_find_any_ne_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vfaebs
  vbc = vec_find_any_ne_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vfaebs
  vbc = vec_find_any_ne_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vfaebs
  vbs = vec_find_any_ne_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vfaehs
  vbs = vec_find_any_ne_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vfaehs
  vbs = vec_find_any_ne_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 12)
  // CHECK-ASM: vfaehs
  vbi = vec_find_any_ne_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vfaefs
  vbi = vec_find_any_ne_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vfaefs
  vbi = vec_find_any_ne_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 12)
  // CHECK-ASM: vfaefs

  vsc = vec_find_any_ne_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaeb
  vuc = vec_find_any_ne_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaeb
  vuc = vec_find_any_ne_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaeb
  vss = vec_find_any_ne_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaeh
  vus = vec_find_any_ne_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaeh
  vus = vec_find_any_ne_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaeh
  vsi = vec_find_any_ne_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaef
  vui = vec_find_any_ne_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaef
  vui = vec_find_any_ne_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfaef(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaef

  vsc = vec_find_any_ne_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaebs
  vuc = vec_find_any_ne_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaebs
  vuc = vec_find_any_ne_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaebs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaebs
  vss = vec_find_any_ne_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaehs
  vus = vec_find_any_ne_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaehs
  vus = vec_find_any_ne_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaehs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaehs
  vsi = vec_find_any_ne_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaefs
  vui = vec_find_any_ne_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaefs
  vui = vec_find_any_ne_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaefs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaefs

  vsc = vec_find_any_ne_or_0_idx(vsc, vsc);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezb
  vuc = vec_find_any_ne_or_0_idx(vuc, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezb
  vuc = vec_find_any_ne_or_0_idx(vbc, vbc);
  // CHECK: call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezb
  vss = vec_find_any_ne_or_0_idx(vss, vss);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezh
  vus = vec_find_any_ne_or_0_idx(vus, vus);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezh
  vus = vec_find_any_ne_or_0_idx(vbs, vbs);
  // CHECK: call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezh
  vsi = vec_find_any_ne_or_0_idx(vsi, vsi);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezf
  vui = vec_find_any_ne_or_0_idx(vui, vui);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezf
  vui = vec_find_any_ne_or_0_idx(vbi, vbi);
  // CHECK: call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezf

  vsc = vec_find_any_ne_or_0_idx_cc(vsc, vsc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezbs
  vuc = vec_find_any_ne_or_0_idx_cc(vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezbs
  vuc = vec_find_any_ne_or_0_idx_cc(vbc, vbc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vfaezbs(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezbs
  vss = vec_find_any_ne_or_0_idx_cc(vss, vss, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezhs
  vus = vec_find_any_ne_or_0_idx_cc(vus, vus, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezhs
  vus = vec_find_any_ne_or_0_idx_cc(vbs, vbs, &cc);
  // CHECK: call { <8 x i16>, i32 } @llvm.s390.vfaezhs(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezhs
  vsi = vec_find_any_ne_or_0_idx_cc(vsi, vsi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezfs
  vui = vec_find_any_ne_or_0_idx_cc(vui, vui, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezfs
  vui = vec_find_any_ne_or_0_idx_cc(vbi, vbi, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfaezfs(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 8)
  // CHECK-ASM: vfaezfs
}

void test_float(void) {
  // CHECK-ASM-LABEL: test_float

  vd = vec_abs(vd);
  // CHECK: call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vflpdb

  vd = vec_nabs(vd);
  // CHECK: [[ABS:%[^ ]+]] = tail call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  // CHECK-NEXT: fneg <2 x double> [[ABS]]
  // CHECK-ASM: vflndb

  vd = vec_madd(vd, vd, vd);
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfmadb
  vd = vec_msub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK-ASM: vfmsdb
  vd = vec_sqrt(vd);
  // CHECK: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfsqdb

  vd = vec_ld2f(cptrf);
  // CHECK: [[VAL:%[^ ]+]] = load <2 x float>, ptr %{{.*}}
  // CHECK: fpext <2 x float> [[VAL]] to <2 x double>
  // (emulated)
  vec_st2f(vd, ptrf);
  // CHECK: [[VAL:%[^ ]+]] = fptrunc <2 x double> %{{.*}} to <2 x float>
  // CHECK: store <2 x float> [[VAL]], ptr %{{.*}}
  // (emulated)

  vd = vec_ctd(vsl, 0);
  // CHECK: sitofp <2 x i64> %{{.*}} to <2 x double>
  // (emulated)
  vd = vec_ctd(vul, 0);
  // CHECK: uitofp <2 x i64> %{{.*}} to <2 x double>
  // (emulated)
  vd = vec_ctd(vsl, 1);
  // CHECK: [[VAL:%[^ ]+]] = sitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK: fmul <2 x double> [[VAL]], <double 5.000000e-01, double 5.000000e-01>
  // (emulated)
  vd = vec_ctd(vul, 1);
  // CHECK: [[VAL:%[^ ]+]] = uitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK: fmul <2 x double> [[VAL]], <double 5.000000e-01, double 5.000000e-01>
  // (emulated)
  vd = vec_ctd(vsl, 31);
  // CHECK: [[VAL:%[^ ]+]] = sitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK: fmul <2 x double> [[VAL]], <double 0x3E00000000000000, double 0x3E00000000000000>
  // (emulated)
  vd = vec_ctd(vul, 31);
  // CHECK: [[VAL:%[^ ]+]] = uitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK: fmul <2 x double> [[VAL]], <double 0x3E00000000000000, double 0x3E00000000000000>
  // (emulated)

  vsl = vec_ctsl(vd, 0);
  // CHECK: fptosi <2 x double> %{{.*}} to <2 x i64>
  // (emulated)
  vul = vec_ctul(vd, 0);
  // CHECK: fptoui <2 x double> %{{.*}} to <2 x i64>
  // (emulated)
  vsl = vec_ctsl(vd, 1);
  // CHECK: [[VAL:%[^ ]+]] = fmul <2 x double> %{{.*}}, <double 2.000000e+00, double 2.000000e+00>
  // CHECK: fptosi <2 x double> [[VAL]] to <2 x i64>
  // (emulated)
  vul = vec_ctul(vd, 1);
  // CHECK: [[VAL:%[^ ]+]] = fmul <2 x double> %{{.*}}, <double 2.000000e+00, double 2.000000e+00>
  // CHECK: fptoui <2 x double> [[VAL]] to <2 x i64>
  // (emulated)
  vsl = vec_ctsl(vd, 31);
  // CHECK: [[VAL:%[^ ]+]] = fmul <2 x double> %{{.*}}, <double 0x41E0000000000000, double 0x41E0000000000000>
  // CHECK: fptosi <2 x double> [[VAL]] to <2 x i64>
  // (emulated)
  vul = vec_ctul(vd, 31);
  // CHECK: [[VAL:%[^ ]+]] = fmul <2 x double> %{{.*}}, <double 0x41E0000000000000, double 0x41E0000000000000>
  // CHECK: fptoui <2 x double> [[VAL]] to <2 x i64>
  // (emulated)

  vd = vec_double(vsl);
  // CHECK: sitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK-ASM: vcdgb
  vd = vec_double(vul);
  // CHECK: uitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK-ASM: vcdlgb

  vsl = vec_signed(vd);
  // CHECK: fptosi <2 x double> %{{.*}} to <2 x i64>
  // CHECK-ASM: vcgdb
  vul = vec_unsigned(vd);
  // CHECK: fptoui <2 x double> %{{.*}} to <2 x i64>
  // CHECK-ASM: vclgdb

  vd = vec_roundp(vd);
  // CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_ceil(vd);
  // CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_roundm(vd);
  // CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_floor(vd);
  // CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_roundz(vd);
  // CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_trunc(vd);
  // CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_roundc(vd);
  // CHECK: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 0
  vd = vec_rint(vd);
  // CHECK: call <2 x double> @llvm.rint.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 0, 0
  vd = vec_round(vd);
  // CHECK: call <2 x double> @llvm.s390.vfidb(<2 x double> %{{.*}}, i32 4, i32 4)
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 4

  vbl = vec_fp_test_data_class(vd, 0, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 0)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, 4095, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 4095)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_ZERO_P, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 2048)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_ZERO_N, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 1024)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_ZERO, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 3072)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_NORMAL_P, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 512)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_NORMAL_N, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 256)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_NORMAL, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 768)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_SUBNORMAL_P, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 128)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_SUBNORMAL_N, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 64)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_SUBNORMAL, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 192)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_INFINITY_P, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 32)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_INFINITY_N, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 16)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_INFINITY, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 48)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_QNAN_P, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 8)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_QNAN_N, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 4)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_QNAN, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 12)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_SNAN_P, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 2)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_SNAN_N, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 1)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_SNAN, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 3)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_NAN, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, __VEC_CLASS_FP_NOT_NORMAL, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 3327)
  // CHECK-ASM: vftcidb
}
