// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-linux-gnu \
// RUN: -O -fzvector -fno-lax-vector-conversions \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s

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
volatile vector float vf;
volatile vector double vd;

volatile signed char sc;
volatile signed short ss;
volatile signed int si;
volatile signed long long sl;
volatile unsigned char uc;
volatile unsigned short us;
volatile unsigned int ui;
volatile unsigned long long ul;
volatile float f;
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
  f = vec_extract(vf, idx);
  // CHECK: extractelement <4 x float> %{{.*}}, i32 %{{.*}}
  d = vec_extract(vd, idx);
  // CHECK: extractelement <2 x double> %{{.*}}, i32 %{{.*}}

  vf = vec_insert(d, vf, idx);
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 %{{.*}}
  vd = vec_insert(f, vd, idx);
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 %{{.*}}

  vf = vec_promote(f, idx);
  // CHECK: insertelement <4 x float> undef, float %{{.*}}, i32 %{{.*}}
  vd = vec_promote(d, idx);
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 %{{.*}}

  vf = vec_insert_and_zero(cptrf);
  // CHECK: insertelement <4 x float> <float undef, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %{{.*}}, i32 0
  vd = vec_insert_and_zero(cptrd);
  // CHECK: insertelement <2 x double> <double undef, double 0.000000e+00>, double %{{.*}}, i32 0

  vf = vec_perm(vf, vf, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vd = vec_perm(vd, vd, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vul = vec_bperm_u128(vuc, vuc);
  // CHECK: call <2 x i64> @llvm.s390.vbperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vf = vec_sel(vf, vf, vui);
  vf = vec_sel(vf, vf, vbi);
  vd = vec_sel(vd, vd, vul);
  vd = vec_sel(vd, vd, vbl);

  vf = vec_gather_element(vf, vui, cptrf, 0);
  vf = vec_gather_element(vf, vui, cptrf, 1);
  vf = vec_gather_element(vf, vui, cptrf, 2);
  vf = vec_gather_element(vf, vui, cptrf, 3);
  vd = vec_gather_element(vd, vul, cptrd, 0);
  vd = vec_gather_element(vd, vul, cptrd, 1);

  vec_scatter_element(vf, vui, ptrf, 0);
  vec_scatter_element(vf, vui, ptrf, 1);
  vec_scatter_element(vf, vui, ptrf, 2);
  vec_scatter_element(vf, vui, ptrf, 3);
  vec_scatter_element(vd, vul, ptrd, 0);
  vec_scatter_element(vd, vul, ptrd, 1);

  vf = vec_xl(idx, cptrf);
  vd = vec_xl(idx, cptrd);

  vec_xst(vf, idx, ptrf);
  vec_xst(vd, idx, ptrd);

  vd = vec_load_bndry(cptrd, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 0)
  vf = vec_load_bndry(cptrf, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 0)
  vf = vec_load_bndry(cptrf, 128);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 1)
  vf = vec_load_bndry(cptrf, 256);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 2)
  vf = vec_load_bndry(cptrf, 512);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 3)
  vf = vec_load_bndry(cptrf, 1024);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 4)
  vf = vec_load_bndry(cptrf, 2048);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 5)
  vf = vec_load_bndry(cptrf, 4096);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 6)

  vf = vec_load_len(cptrf, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, i8* %{{.*}})
  vd = vec_load_len(cptrd, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, i8* %{{.*}})

  vec_store_len(vf, ptrf, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})
  vec_store_len(vd, ptrd, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})

  vuc = vec_load_len_r(cptruc, idx);
  // CHECK: call <16 x i8> @llvm.s390.vlrl(i32 %{{.*}}, i8* %{{.*}})

  vec_store_len_r(vuc, ptruc, idx);
  // CHECK: call void @llvm.s390.vstrl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})

  vf = vec_splat(vf, 0);
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> undef, <4 x i32> zeroinitializer
  vf = vec_splat(vf, 1);
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  vd = vec_splat(vd, 0);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> undef, <2 x i32> zeroinitializer
  vd = vec_splat(vd, 1);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> undef, <2 x i32> <i32 1, i32 1>

  vf = vec_splats(f);
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> undef, <4 x i32> zeroinitializer
  vd = vec_splats(d);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> undef, <2 x i32> zeroinitializer

  vf = vec_mergeh(vf, vf);
  // shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  vd = vec_mergeh(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>

  vf = vec_mergel(vf, vf);
  // shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <i32 2, i32 6, i32 3, i32 7>
  vd = vec_mergel(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <i32 1, i32 3>
}

void test_compare(void) {
  vbi = vec_cmpeq(vf, vf);
  // CHECK: fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  vbl = vec_cmpeq(vd, vd);
  // CHECK: fcmp oeq <2 x double> %{{.*}}, %{{.*}}

  vbi = vec_cmpge(vf, vf);
  // CHECK: fcmp oge <4 x float> %{{.*}}, %{{.*}}
  vbl = vec_cmpge(vd, vd);
  // CHECK: fcmp oge <2 x double> %{{.*}}, %{{.*}}

  vbi = vec_cmpgt(vf, vf);
  // CHECK: fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  vbl = vec_cmpgt(vd, vd);
  // CHECK: fcmp ogt <2 x double> %{{.*}}, %{{.*}}

  vbi = vec_cmple(vf, vf);
  // CHECK: fcmp ole <4 x float> %{{.*}}, %{{.*}}
  vbl = vec_cmple(vd, vd);
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}

  vbi = vec_cmplt(vf, vf);
  // CHECK: fcmp olt <4 x float> %{{.*}}, %{{.*}}
  vbl = vec_cmplt(vd, vd);
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}

  idx = vec_all_eq(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_eq(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_ne(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_ne(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_ge(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_ge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_gt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_gt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_le(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_le(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_lt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_lt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_nge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  idx = vec_all_nge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_ngt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_ngt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_nle(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_nle(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_nlt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_all_nlt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_all_nan(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  idx = vec_all_nan(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)

  idx = vec_all_numeric(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  idx = vec_all_numeric(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)

  idx = vec_any_eq(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_eq(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_ne(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_ne(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_ge(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_ge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_gt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_gt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_le(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_le(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_lt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_lt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_nge(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_nge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_ngt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_ngt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_nle(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_nle(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_nlt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  idx = vec_any_nlt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  idx = vec_any_nan(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  idx = vec_any_nan(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)

  idx = vec_any_numeric(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  idx = vec_any_numeric(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
}

void test_integer(void) {
  vf = vec_andc(vf, vf);
  vd = vec_andc(vd, vd);

  vf = vec_nor(vf, vf);
  vd = vec_nor(vd, vd);

  vsc = vec_nand(vsc, vsc);
  vuc = vec_nand(vuc, vuc);
  vbc = vec_nand(vbc, vbc);
  vss = vec_nand(vss, vss);
  vus = vec_nand(vus, vus);
  vbs = vec_nand(vbs, vbs);
  vsi = vec_nand(vsi, vsi);
  vui = vec_nand(vui, vui);
  vbi = vec_nand(vbi, vbi);
  vsl = vec_nand(vsl, vsl);
  vul = vec_nand(vul, vul);
  vbl = vec_nand(vbl, vbl);
  vf = vec_nand(vf, vf);
  vd = vec_nand(vd, vd);

  vsc = vec_orc(vsc, vsc);
  vuc = vec_orc(vuc, vuc);
  vbc = vec_orc(vbc, vbc);
  vss = vec_orc(vss, vss);
  vus = vec_orc(vus, vus);
  vbs = vec_orc(vbs, vbs);
  vsi = vec_orc(vsi, vsi);
  vui = vec_orc(vui, vui);
  vbi = vec_orc(vbi, vbi);
  vsl = vec_orc(vsl, vsl);
  vul = vec_orc(vul, vul);
  vbl = vec_orc(vbl, vbl);
  vf = vec_orc(vf, vf);
  vd = vec_orc(vd, vd);

  vsc = vec_eqv(vsc, vsc);
  vuc = vec_eqv(vuc, vuc);
  vbc = vec_eqv(vbc, vbc);
  vss = vec_eqv(vss, vss);
  vus = vec_eqv(vus, vus);
  vbs = vec_eqv(vbs, vbs);
  vsi = vec_eqv(vsi, vsi);
  vui = vec_eqv(vui, vui);
  vbi = vec_eqv(vbi, vbi);
  vsl = vec_eqv(vsl, vsl);
  vul = vec_eqv(vul, vul);
  vbl = vec_eqv(vbl, vbl);
  vf = vec_eqv(vf, vf);
  vd = vec_eqv(vd, vd);

  vf = vec_slb(vf, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vf = vec_slb(vf, vui);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vd = vec_slb(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vd = vec_slb(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vf = vec_sld(vf, vf, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vf = vec_sld(vf, vf, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  vd = vec_sld(vd, vd, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vd = vec_sld(vd, vd, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)

  vf = vec_srab(vf, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vf = vec_srab(vf, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vd = vec_srab(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vd = vec_srab(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vf = vec_srb(vf, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vf = vec_srb(vf, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vd = vec_srb(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  vd = vec_srb(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  idx = vec_test_mask(vf, vui);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  idx = vec_test_mask(vd, vul);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})

  vuc = vec_msum_u128(vul, vul, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  vuc = vec_msum_u128(vul, vul, vuc, 4);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  vuc = vec_msum_u128(vul, vul, vuc, 8);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  vuc = vec_msum_u128(vul, vul, vuc, 12);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
}

void test_float(void) {
  vf = vec_abs(vf);
  // CHECK: call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  vd = vec_abs(vd);
  // CHECK: call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})

  vf = vec_nabs(vf);
  // CHECK: [[ABS:%[^ ]+]] = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  // CHECK-NEXT: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, [[ABS]]
  vd = vec_nabs(vd);
  // CHECK: [[ABS:%[^ ]+]] = tail call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  // CHECK-NEXT: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, [[ABS]]

  vf = vec_max(vf, vf);
  // CHECK: call <4 x float> @llvm.s390.vfmaxsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  vd = vec_max(vd, vd);
  // CHECK: call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)

  vf = vec_min(vf, vf);
  // CHECK: call <4 x float> @llvm.s390.vfminsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  vd = vec_min(vd, vd);
  // CHECK: call <2 x double> @llvm.s390.vfmindb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)

  vf = vec_madd(vf, vf, vf);
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  vd = vec_madd(vd, vd, vd);
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})

  vf = vec_msub(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  vd = vec_msub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])

  vf = vec_nmadd(vf, vf, vf);
  // CHECK: [[RES:%[^ ]+]] = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, [[RES]]
  vd = vec_nmadd(vd, vd, vd);
  // CHECK: [[RES:%[^ ]+]] = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, [[RES]]

  vf = vec_nmsub(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  // CHECK: fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, [[RES]]
  vd = vec_nmsub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK: fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, [[RES]]

  vf = vec_sqrt(vf);
  // CHECK: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.*}})
  vd = vec_sqrt(vd);
  // CHECK: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{.*}})

  vd = vec_doublee(vf);
  // CHECK: fpext <2 x float> %{{.*}} to <2 x double>
  vf = vec_floate(vd);
  // CHECK: fptrunc <2 x double> %{{.*}} to <2 x float>

  vd = vec_double(vsl);
  // CHECK: sitofp <2 x i64> %{{.*}} to <2 x double>
  vd = vec_double(vul);
  // CHECK: uitofp <2 x i64> %{{.*}} to <2 x double>

  vsl = vec_signed(vd);
  // CHECK: fptosi <2 x double> %{{.*}} to <2 x i64>
  vul = vec_unsigned(vd);
  // CHECK: fptoui <2 x double> %{{.*}} to <2 x i64>

  vf = vec_roundp(vf);
  // CHECK: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{.*}})
  vf = vec_ceil(vf);
  // CHECK: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{.*}})
  vd = vec_roundp(vd);
  // CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{.*}})
  vd = vec_ceil(vd);
  // CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{.*}})

  vf = vec_roundm(vf);
  // CHECK: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})
  vf = vec_floor(vf);
  // CHECK: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})
  vd = vec_roundm(vd);
  // CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{.*}})
  vd = vec_floor(vd);
  // CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{.*}})

  vf = vec_roundz(vf);
  // CHECK: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{.*}})
  vf = vec_trunc(vf);
  // CHECK: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{.*}})
  vd = vec_roundz(vd);
  // CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{.*}})
  vd = vec_trunc(vd);
  // CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{.*}})

  vf = vec_roundc(vf);
  // CHECK: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %{{.*}})
  vd = vec_roundc(vd);
  // CHECK: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %{{.*}})

  vf = vec_rint(vf);
  // CHECK: call <4 x float> @llvm.rint.v4f32(<4 x float> %{{.*}})
  vd = vec_rint(vd);
  // CHECK: call <2 x double> @llvm.rint.v2f64(<2 x double> %{{.*}})

  vf = vec_round(vf);
  // CHECK: call <4 x float> @llvm.s390.vfisb(<4 x float> %{{.*}}, i32 4, i32 4)
  vd = vec_round(vd);
  // CHECK: call <2 x double> @llvm.s390.vfidb(<2 x double> %{{.*}}, i32 4, i32 4)

  vbi = vec_fp_test_data_class(vf, 0, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 0)
  vbi = vec_fp_test_data_class(vf, 4095, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 4095)
  vbl = vec_fp_test_data_class(vd, 0, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 0)
  vbl = vec_fp_test_data_class(vd, 4095, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 4095)
}
