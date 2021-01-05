// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-linux-gnu \
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
  // CHECK-ASM-LABEL: test_core
  vector float vf2;
  vector double vd2;

  f = vec_extract(vf, 0);
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK-ASM: vstef
  f = vec_extract(vf, idx);
  // CHECK: extractelement <4 x float> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvf
  d = vec_extract(vd, 0);
  // CHECK: extractelement <2 x double> %{{.*}}, i32 0
  // CHECK-ASM: vsteg
  d = vec_extract(vd, idx);
  // CHECK: extractelement <2 x double> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvg

  vf2 = vf;
  vf = vec_insert(f, vf2, 0);
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  // CHECK-ASM: vlef
  vf = vec_insert(0.0f, vf, 1);
  // CHECK: insertelement <4 x float> %{{.*}}, float 0.000000e+00, i32 1
  // CHECK-ASM: vleif %{{.*}}, 0, 1
  vf = vec_insert(f, vf, idx);
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgf
  vd2 = vd;
  vd = vec_insert(d, vd2, 0);
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  // CHECK-ASM: vleg
  vd = vec_insert(0.0, vd, 1);
  // CHECK: insertelement <2 x double> %{{.*}}, double 0.000000e+00, i32 1
  // CHECK-ASM: vleig %{{.*}}, 0, 1
  vd = vec_insert(d, vd, idx);
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg

  vf = vec_promote(f, idx);
  // CHECK: insertelement <4 x float> undef, float %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgf
  vd = vec_promote(d, idx);
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg

  vf = vec_insert_and_zero(cptrf);
  // CHECK: insertelement <4 x float> <float 0.000000e+00, float poison, float 0.000000e+00, float 0.000000e+00>, float %{{.*}}, i32 1
  // CHECK-ASM: vllezf
  vd = vec_insert_and_zero(cptrd);
  // CHECK: insertelement <2 x double> <double poison, double 0.000000e+00>, double %{{.*}}, i32 0
  // CHECK-ASM: vllezg

  vf = vec_perm(vf, vf, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm
  vd = vec_perm(vd, vd, vuc);
  // CHECK: call <16 x i8> @llvm.s390.vperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vperm

  vul = vec_bperm_u128(vuc, vuc);
  // CHECK: call <2 x i64> @llvm.s390.vbperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vbperm

  vf = vec_revb(vf);
  // CHECK-ASM: vperm
  vd = vec_revb(vd);
  // CHECK-ASM: vperm

  vf = vec_reve(vf);
  // CHECK-ASM: vperm
  vd = vec_reve(vd);
  // CHECK-ASM: {{vperm|vpdi}}

  vf = vec_sel(vf, vf, vui);
  // CHECK-ASM: vsel
  vf = vec_sel(vf, vf, vbi);
  // CHECK-ASM: vsel
  vd = vec_sel(vd, vd, vul);
  // CHECK-ASM: vsel
  vd = vec_sel(vd, vd, vbl);
  // CHECK-ASM: vsel

  vf = vec_gather_element(vf, vui, cptrf, 0);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vf = vec_gather_element(vf, vui, cptrf, 1);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vf = vec_gather_element(vf, vui, cptrf, 2);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vf = vec_gather_element(vf, vui, cptrf, 3);
  // CHECK-ASM: vgef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vd = vec_gather_element(vd, vul, cptrd, 0);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vd = vec_gather_element(vd, vul, cptrd, 1);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 1

  vec_scatter_element(vf, vui, ptrf, 0);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vf, vui, ptrf, 1);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 1
  vec_scatter_element(vf, vui, ptrf, 2);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 2
  vec_scatter_element(vf, vui, ptrf, 3);
  // CHECK-ASM: vscef %{{.*}}, 0(%{{.*}},%{{.*}}), 3
  vec_scatter_element(vd, vul, ptrd, 0);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vd, vul, ptrd, 1);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 1

  vf = vec_xl(idx, cptrf);
  // CHECK-ASM: vl
  vd = vec_xl(idx, cptrd);
  // CHECK-ASM: vl

  vec_xst(vf, idx, ptrf);
  // CHECK-ASM: vst
  vec_xst(vd, idx, ptrd);
  // CHECK-ASM: vst

  vd = vec_load_bndry(cptrd, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vf = vec_load_bndry(cptrf, 64);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 0)
  // CHECK-ASM: vlbb
  vf = vec_load_bndry(cptrf, 128);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 1)
  // CHECK-ASM: vlbb
  vf = vec_load_bndry(cptrf, 256);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 2)
  // CHECK-ASM: vlbb
  vf = vec_load_bndry(cptrf, 512);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 3)
  // CHECK-ASM: vlbb
  vf = vec_load_bndry(cptrf, 1024);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 4)
  // CHECK-ASM: vlbb
  vf = vec_load_bndry(cptrf, 2048);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 5)
  // CHECK-ASM: vlbb
  vf = vec_load_bndry(cptrf, 4096);
  // CHECK: call <16 x i8> @llvm.s390.vlbb(i8* %{{.*}}, i32 6)
  // CHECK-ASM: vlbb

  vf = vec_load_len(cptrf, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, i8* %{{.*}})
  // CHECK-ASM: vll
  vd = vec_load_len(cptrd, idx);
  // CHECK: call <16 x i8> @llvm.s390.vll(i32 %{{.*}}, i8* %{{.*}})
  // CHECK-ASM: vll

  vec_store_len(vf, ptrf, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})
  // CHECK-ASM: vstl
  vec_store_len(vd, ptrd, idx);
  // CHECK: call void @llvm.s390.vstl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})
  // CHECK-ASM: vstl

  vuc = vec_load_len_r(cptruc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vlrl(i32 0, i8* %{{.*}})
  // CHECK-ASM: vlrl %{{.*}}, 0(%{{.*}}), 0
  vuc = vec_load_len_r(cptruc, idx);
  // CHECK: call <16 x i8> @llvm.s390.vlrl(i32 %{{.*}}, i8* %{{.*}})
  // CHECK-ASM: vlrlr

  vec_store_len_r(vuc, ptruc, 0);
  // CHECK: call void @llvm.s390.vstrl(<16 x i8> %{{.*}}, i32 0, i8* %{{.*}})
  // CHECK-ASM: vstrl %{{.*}}, 0(%{{.*}}), 0
  vec_store_len_r(vuc, ptruc, idx);
  // CHECK: call void @llvm.s390.vstrl(<16 x i8> %{{.*}}, i32 %{{.*}}, i8* %{{.*}})
  // CHECK-ASM: vstrlr

  vf = vec_splat(vf, 0);
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> zeroinitializer
  // CHECK-ASM: vrepf
  vf = vec_splat(vf, 1);
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  // CHECK-ASM: vrepf
  vd = vec_splat(vd, 0);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vrepg
  vd = vec_splat(vd, 1);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  // CHECK-ASM: vrepg

  vf = vec_splats(f);
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> zeroinitializer
  // CHECK-ASM: vlrepf
  vd = vec_splats(d);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vlrepg

  vf = vec_mergeh(vf, vf);
  // shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK-ASM: vmrhf
  vd = vec_mergeh(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK-ASM: vmrhg

  vf = vec_mergel(vf, vf);
  // shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <i32 2, i32 6, i32 3, i32 7>
  // CHECK-ASM: vmrlf
  vd = vec_mergel(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <i32 1, i32 3>
  // CHECK-ASM: vmrlg
}

void test_compare(void) {
  // CHECK-ASM-LABEL: test_compare

  vbi = vec_cmpeq(vf, vf);
  // CHECK: fcmp oeq <4 x float> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfcesb
  vbl = vec_cmpeq(vd, vd);
  // CHECK: fcmp oeq <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfcedb

  vbi = vec_cmpge(vf, vf);
  // CHECK: fcmp oge <4 x float> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchesb
  vbl = vec_cmpge(vd, vd);
  // CHECK: fcmp oge <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchedb

  vbi = vec_cmpgt(vf, vf);
  // CHECK: fcmp ogt <4 x float> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchsb
  vbl = vec_cmpgt(vd, vd);
  // CHECK: fcmp ogt <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchdb

  vbi = vec_cmple(vf, vf);
  // CHECK: fcmp ole <4 x float> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchesb
  vbl = vec_cmple(vd, vd);
  // CHECK: fcmp ole <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchedb

  vbi = vec_cmplt(vf, vf);
  // CHECK: fcmp olt <4 x float> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchsb
  vbl = vec_cmplt(vd, vd);
  // CHECK: fcmp olt <2 x double> %{{.*}}, %{{.*}}
  // CHECK-ASM: vfchdb

  idx = vec_all_eq(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfcesbs
  idx = vec_all_eq(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_all_ne(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfcesbs
  idx = vec_all_ne(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_all_ge(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_all_ge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_all_gt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_all_gt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_all_le(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_all_le(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_all_lt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_all_lt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_all_nge(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_all_nge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_all_ngt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_all_ngt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_all_nle(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_all_nle(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_all_nlt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_all_nlt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_all_nan(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  // CHECK-ASM: vftcisb
  idx = vec_all_nan(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb

  idx = vec_all_numeric(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  // CHECK-ASM: vftcisb
  idx = vec_all_numeric(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb

  idx = vec_any_eq(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfcesbs
  idx = vec_any_eq(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_any_ne(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfcesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfcesbs
  idx = vec_any_ne(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_any_ge(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_any_ge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_any_gt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_any_gt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_le(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_any_le(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_any_lt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_any_lt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_nge(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_any_nge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_any_ngt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_any_ngt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_nle(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchesbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchesbs
  idx = vec_any_nle(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_any_nlt(vf, vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vfchsbs(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfchsbs
  idx = vec_any_nlt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_nan(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  // CHECK-ASM: vftcisb
  idx = vec_any_nan(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb

  idx = vec_any_numeric(vf);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 15)
  // CHECK-ASM: vftcisb
  idx = vec_any_numeric(vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 15)
  // CHECK-ASM: vftcidb
}

void test_integer(void) {
  // CHECK-ASM-LABEL: test_integer

  vf = vec_andc(vf, vf);
  // CHECK-ASM: vnc
  vd = vec_andc(vd, vd);
  // CHECK-ASM: vnc

  vf = vec_nor(vf, vf);
  // CHECK-ASM: vno
  vd = vec_nor(vd, vd);
  // CHECK-ASM: vno

  vsc = vec_nand(vsc, vsc);
  // CHECK-ASM: vnn
  vuc = vec_nand(vuc, vuc);
  // CHECK-ASM: vnn
  vbc = vec_nand(vbc, vbc);
  // CHECK-ASM: vnn
  vss = vec_nand(vss, vss);
  // CHECK-ASM: vnn
  vus = vec_nand(vus, vus);
  // CHECK-ASM: vnn
  vbs = vec_nand(vbs, vbs);
  // CHECK-ASM: vnn
  vsi = vec_nand(vsi, vsi);
  // CHECK-ASM: vnn
  vui = vec_nand(vui, vui);
  // CHECK-ASM: vnn
  vbi = vec_nand(vbi, vbi);
  // CHECK-ASM: vnn
  vsl = vec_nand(vsl, vsl);
  // CHECK-ASM: vnn
  vul = vec_nand(vul, vul);
  // CHECK-ASM: vnn
  vbl = vec_nand(vbl, vbl);
  // CHECK-ASM: vnn
  vf = vec_nand(vf, vf);
  // CHECK-ASM: vnn
  vd = vec_nand(vd, vd);
  // CHECK-ASM: vnn

  vsc = vec_orc(vsc, vsc);
  // CHECK-ASM: voc
  vuc = vec_orc(vuc, vuc);
  // CHECK-ASM: voc
  vbc = vec_orc(vbc, vbc);
  // CHECK-ASM: voc
  vss = vec_orc(vss, vss);
  // CHECK-ASM: voc
  vus = vec_orc(vus, vus);
  // CHECK-ASM: voc
  vbs = vec_orc(vbs, vbs);
  // CHECK-ASM: voc
  vsi = vec_orc(vsi, vsi);
  // CHECK-ASM: voc
  vui = vec_orc(vui, vui);
  // CHECK-ASM: voc
  vbi = vec_orc(vbi, vbi);
  // CHECK-ASM: voc
  vsl = vec_orc(vsl, vsl);
  // CHECK-ASM: voc
  vul = vec_orc(vul, vul);
  // CHECK-ASM: voc
  vbl = vec_orc(vbl, vbl);
  // CHECK-ASM: voc
  vf = vec_orc(vf, vf);
  // CHECK-ASM: voc
  vd = vec_orc(vd, vd);
  // CHECK-ASM: voc

  vsc = vec_eqv(vsc, vsc);
  // CHECK-ASM: vnx
  vuc = vec_eqv(vuc, vuc);
  // CHECK-ASM: vnx
  vbc = vec_eqv(vbc, vbc);
  // CHECK-ASM: vnx
  vss = vec_eqv(vss, vss);
  // CHECK-ASM: vnx
  vus = vec_eqv(vus, vus);
  // CHECK-ASM: vnx
  vbs = vec_eqv(vbs, vbs);
  // CHECK-ASM: vnx
  vsi = vec_eqv(vsi, vsi);
  // CHECK-ASM: vnx
  vui = vec_eqv(vui, vui);
  // CHECK-ASM: vnx
  vbi = vec_eqv(vbi, vbi);
  // CHECK-ASM: vnx
  vsl = vec_eqv(vsl, vsl);
  // CHECK-ASM: vnx
  vul = vec_eqv(vul, vul);
  // CHECK-ASM: vnx
  vbl = vec_eqv(vbl, vbl);
  // CHECK-ASM: vnx
  vf = vec_eqv(vf, vf);
  // CHECK-ASM: vnx
  vd = vec_eqv(vd, vd);
  // CHECK-ASM: vnx

  vuc = vec_popcnt(vsc);
  // CHECK: call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %{{.*}})
  // CHECK-ASM: vpopctb
  vuc = vec_popcnt(vuc);
  // CHECK: call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %{{.*}})
  // CHECK-ASM: vpopctb
  vus = vec_popcnt(vss);
  // CHECK: call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %{{.*}})
  // CHECK-ASM: vpopcth
  vus = vec_popcnt(vus);
  // CHECK: call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %{{.*}})
  // CHECK-ASM: vpopcth
  vui = vec_popcnt(vsi);
  // CHECK: call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %{{.*}})
  // CHECK-ASM: vpopctf
  vui = vec_popcnt(vui);
  // CHECK: call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %{{.*}})
  // CHECK-ASM: vpopctf
  vul = vec_popcnt(vsl);
  // CHECK: call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %{{.*}})
  // CHECK-ASM: vpopctg
  vul = vec_popcnt(vul);
  // CHECK: call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %{{.*}})
  // CHECK-ASM: vpopctg

  vf = vec_slb(vf, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vf = vec_slb(vf, vui);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vd = vec_slb(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb
  vd = vec_slb(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vslb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vslb

  vf = vec_sld(vf, vf, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vf = vec_sld(vf, vf, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb
  vd = vec_sld(vd, vd, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsldb
  vd = vec_sld(vd, vd, 15);
  // CHECK: call <16 x i8> @llvm.s390.vsldb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 15)
  // CHECK-ASM: vsldb

  vf = vec_srab(vf, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vf = vec_srab(vf, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vd = vec_srab(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab
  vd = vec_srab(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrab

  vf = vec_srb(vf, vsi);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vf = vec_srb(vf, vui);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vd = vec_srb(vd, vsl);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb
  vd = vec_srb(vd, vul);
  // CHECK: call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vsrlb

  idx = vec_test_mask(vf, vui);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm
  idx = vec_test_mask(vd, vul);
  // CHECK: call i32 @llvm.s390.vtm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vtm

  vuc = vec_msum_u128(vul, vul, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vmslg
  vuc = vec_msum_u128(vul, vul, vuc, 4);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 4)
  // CHECK-ASM: vmslg
  vuc = vec_msum_u128(vul, vul, vuc, 8);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 8)
  // CHECK-ASM: vmslg
  vuc = vec_msum_u128(vul, vul, vuc, 12);
  // CHECK: call <16 x i8> @llvm.s390.vmslg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <16 x i8> %{{.*}}, i32 12)
  // CHECK-ASM: vmslg
}

void test_float(void) {
  // CHECK-ASM-LABEL: test_float

  vf = vec_abs(vf);
  // CHECK: call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vflpsb
  vd = vec_abs(vd);
  // CHECK: call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vflpdb

  vf = vec_nabs(vf);
  // CHECK: [[ABS:%[^ ]+]] = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  // CHECK-NEXT: fneg <4 x float> [[ABS]]
  // CHECK-ASM: vflnsb
  vd = vec_nabs(vd);
  // CHECK: [[ABS:%[^ ]+]] = tail call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  // CHECK-NEXT: fneg <2 x double> [[ABS]]
  // CHECK-ASM: vflndb

  vf = vec_max(vf, vf);
  // CHECK: call <4 x float> @llvm.s390.vfmaxsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  // CHECK-ASM: vfmaxsb
  vd = vec_max(vd, vd);
  // CHECK: call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)
  // CHECK-ASM: vfmaxdb

  vf = vec_min(vf, vf);
  // CHECK: call <4 x float> @llvm.s390.vfminsb(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  // CHECK-ASM: vfminsb
  vd = vec_min(vd, vd);
  // CHECK: call <2 x double> @llvm.s390.vfmindb(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)
  // CHECK-ASM: vfmindb

  vf = vec_madd(vf, vf, vf);
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfmasb
  vd = vec_madd(vd, vd, vd);
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfmadb

  vf = vec_msub(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  // CHECK-ASM: vfmssb
  vd = vec_msub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK-ASM: vfmsdb

  vf = vec_nmadd(vf, vf, vf);
  // CHECK: [[RES:%[^ ]+]] = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: fneg <4 x float> [[RES]]
  // CHECK-ASM: vfnmasb
  vd = vec_nmadd(vd, vd, vd);
  // CHECK: [[RES:%[^ ]+]] = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: fneg <2 x double> [[RES]]
  // CHECK-ASM: vfnmadb

  vf = vec_nmsub(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  // CHECK: fneg <4 x float> [[RES]]
  // CHECK-ASM: vfnmssb
  vd = vec_nmsub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = tail call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  // CHECK: fneg <2 x double> [[RES]]
  // CHECK-ASM: vfnmsdb

  vf = vec_sqrt(vf);
  // CHECK: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfsqsb
  vd = vec_sqrt(vd);
  // CHECK: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfsqdb

  vd = vec_doublee(vf);
  // CHECK: fpext <2 x float> %{{.*}} to <2 x double>
  // CHECK-ASM: vldeb
  vf = vec_floate(vd);
  // CHECK: fptrunc <2 x double> %{{.*}} to <2 x float>
  // CHECK-ASM: vledb

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

  vf = vec_roundp(vf);
  // CHECK: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 6
  vf = vec_ceil(vf);
  // CHECK: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_roundp(vd);
  // CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_ceil(vd);
  // CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6

  vf = vec_roundm(vf);
  // CHECK: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 7
  vf = vec_floor(vf);
  // CHECK: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_roundm(vd);
  // CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_floor(vd);
  // CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7

  vf = vec_roundz(vf);
  // CHECK: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 5
  vf = vec_trunc(vf);
  // CHECK: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_roundz(vd);
  // CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_trunc(vd);
  // CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5

  vf = vec_roundc(vf);
  // CHECK: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 0
  vd = vec_roundc(vd);
  // CHECK: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 0

  vf = vec_rint(vf);
  // CHECK: call <4 x float> @llvm.rint.v4f32(<4 x float> %{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 0, 0
  vd = vec_rint(vd);
  // CHECK: call <2 x double> @llvm.rint.v2f64(<2 x double> %{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 0, 0

  vf = vec_round(vf);
  // CHECK: call <4 x float> @llvm.s390.vfisb(<4 x float> %{{.*}}, i32 4, i32 4)
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 4
  vd = vec_round(vd);
  // CHECK: call <2 x double> @llvm.s390.vfidb(<2 x double> %{{.*}}, i32 4, i32 4)
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 4

  vbi = vec_fp_test_data_class(vf, 0, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 0)
  // CHECK-ASM: vftcisb
  vbi = vec_fp_test_data_class(vf, 4095, &cc);
  // CHECK: call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %{{.*}}, i32 4095)
  // CHECK-ASM: vftcisb
  vbl = vec_fp_test_data_class(vd, 0, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 0)
  // CHECK-ASM: vftcidb
  vbl = vec_fp_test_data_class(vd, 4095, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 4095)
  // CHECK-ASM: vftcidb
}
