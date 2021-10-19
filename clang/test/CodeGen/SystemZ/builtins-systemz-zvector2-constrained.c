// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed long long vsl;
volatile vector unsigned int vui;
volatile vector unsigned long long vul;
volatile vector bool int vbi;
volatile vector bool long long vbl;
volatile vector float vf;
volatile vector double vd;

volatile float f;
volatile double d;

const float * volatile cptrf;
const double * volatile cptrd;

float * volatile ptrf;
double * volatile ptrd;

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
  // CHECK: insertelement <4 x float> <float 0.000000e+00, float poison, float 0.000000e+00, float 0.000000e+00>, float {{.*}}, i32 1
  // CHECK-ASM: vllezf
  vd = vec_insert_and_zero(cptrd);
  // CHECK: insertelement <2 x double> <double poison, double 0.000000e+00>, double %{{.*}}, i32 0
  // CHECK-ASM: vllezg

  vf = vec_revb(vf);
  // CHECK-ASM: vperm
  vd = vec_revb(vd);
  // CHECK-ASM: vperm

  vf = vec_reve(vf);
  // CHECK-ASM: vperm
  vd = vec_reve(vd);
  // CHECK-ASM: {{vperm|vpdi}}

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
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !{{.*}})
  // CHECK-ASM: vfcesb
  vbl = vec_cmpeq(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq", metadata !{{.*}})
  // CHECK-ASM: vfcedb

  vbi = vec_cmpge(vf, vf);
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oge", metadata !{{.*}})
  // CHECK-ASM: vfkhesb
  vbl = vec_cmpge(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge", metadata !{{.*}})
  // CHECK-ASM: vfkhedb

  vbi = vec_cmpgt(vf, vf);
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ogt", metadata !{{.*}})
  // CHECK-ASM: vfkhsb
  vbl = vec_cmpgt(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt", metadata !{{.*}})
  // CHECK-ASM: vfkhdb

  vbi = vec_cmple(vf, vf);
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole", metadata !{{.*}})
  // CHECK-ASM: vfkhesb
  vbl = vec_cmple(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !{{.*}})
  // CHECK-ASM: vfkhedb

  vbi = vec_cmplt(vf, vf);
  // CHECK: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt", metadata !{{.*}})
  // CHECK-ASM: vfkhsb
  vbl = vec_cmplt(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !{{.*}})
  // CHECK-ASM: vfkhdb

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
  // CHECK: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmasb
  vd = vec_madd(vd, vd, vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadb

  vf = vec_msub(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]], metadata !{{.*}})
  // CHECK-ASM: vfmssb
  vd = vec_msub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]], metadata !{{.*}})
  // CHECK-ASM: vfmsdb

  vf = vec_nmadd(vf, vf, vf);
  // CHECK: [[RES:%[^ ]+]] = tail call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK: fneg <4 x float> [[RES]]
  // CHECK-ASM: vfnmasb
  vd = vec_nmadd(vd, vd, vd);
  // CHECK: [[RES:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK: fneg <2 x double> [[RES]]
  // CHECK-ASM: vfnmadb

  vf = vec_nmsub(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = tail call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]], metadata !{{.*}})
  // CHECK: fneg <4 x float> [[RES]]
  // CHECK-ASM: vfnmssb
  vd = vec_nmsub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]], metadata !{{.*}})
  // CHECK: fneg <2 x double> [[RES]]
  // CHECK-ASM: vfnmsdb

  vf = vec_sqrt(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.sqrt.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfsqsb
  vd = vec_sqrt(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.sqrt.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfsqdb

  vd = vec_doublee(vf);
  // CHECK: call <2 x double> @llvm.experimental.constrained.fpext.v2f64.v2f32(<2 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vldeb
  vf = vec_floate(vd);
  // CHECK: call <2 x float> @llvm.experimental.constrained.fptrunc.v2f32.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vledb

  vd = vec_double(vsl);
  // CHECK: call <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcdgb
  vd = vec_double(vul);
  // CHECK: call <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcdlgb

  vsl = vec_signed(vd);
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcgdb
  vul = vec_unsigned(vd);
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vclgdb

  vf = vec_roundp(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.ceil.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 6
  vf = vec_ceil(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.ceil.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_roundp(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_ceil(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6

  vf = vec_roundm(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.floor.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 7
  vf = vec_floor(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.floor.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_roundm(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.floor.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_floor(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.floor.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7

  vf = vec_roundz(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.trunc.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 5
  vf = vec_trunc(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.trunc.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_roundz(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.trunc.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_trunc(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.trunc.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5

  vf = vec_roundc(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.nearbyint.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 4, 0
  vd = vec_roundc(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.nearbyint.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 0

  vf = vec_rint(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.rint.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfisb %{{.*}}, %{{.*}}, 0, 0
  vd = vec_rint(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
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
