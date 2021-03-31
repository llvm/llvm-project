// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed long long vsl;
volatile vector unsigned long long vul;
volatile vector bool long long vbl;
volatile vector double vd;

volatile double d;

const float * volatile cptrf;
const double * volatile cptrd;

float * volatile ptrf;
double * volatile ptrd;

volatile int idx;

void test_core(void) {
  // CHECK-ASM-LABEL: test_core

  d = vec_extract(vd, idx);
  // CHECK: extractelement <2 x double> %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlgvg

  vd = vec_insert(d, vd, idx);
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg

  vd = vec_promote(d, idx);
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 %{{.*}}
  // CHECK-ASM: vlvgg

  vd = vec_insert_and_zero(cptrd);
  // CHECK: [[ZVEC:%[^ ]+]] = insertelement <2 x double> <double poison, double 0.000000e+00>, double {{.*}}, i32 0
  // CHECK-ASM: vllezg

  vd = vec_revb(vd);
  // CHECK-ASM: vperm

  vd = vec_reve(vd);
  // CHECK-ASM: {{vperm|vpdi}}

  vd = vec_sel(vd, vd, vul);
  // CHECK-ASM: vsel
  vd = vec_sel(vd, vd, vbl);
  // CHECK-ASM: vsel

  vd = vec_gather_element(vd, vul, cptrd, 0);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vd = vec_gather_element(vd, vul, cptrd, 1);
  // CHECK-ASM: vgeg %{{.*}}, 0(%{{.*}},%{{.*}}), 1

  vec_scatter_element(vd, vul, ptrd, 0);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 0
  vec_scatter_element(vd, vul, ptrd, 1);
  // CHECK-ASM: vsceg %{{.*}}, 0(%{{.*}},%{{.*}}), 1

  vd = vec_xl(idx, cptrd);
  // CHECK-ASM-NEXT: lgfrl   %r3, idx
  // CHECK-ASM-NEXT: lgrl    %r4, cptrd
  // CHECK-ASM-NEXT: vl      %v0, 0(%r3,%r4){{$}}
  // CHECK-ASM-NEXT: vst

  vd = vec_xld2(idx, cptrd);
  // CHECK-ASM-NEXT: lgfrl   %r3, idx
  // CHECK-ASM-NEXT: lgrl    %r4, cptrd
  // CHECK-ASM-NEXT: vl      %v0, 0(%r3,%r4){{$}}
  // CHECK-ASM-NEXT: vst

  vec_xst(vd, idx, ptrd);
  // CHECK-ASM-NEXT: vl
  // CHECK-ASM-NEXT: lgfrl   %r3, idx
  // CHECK-ASM-NEXT: lgrl    %r4, ptrd
  // CHECK-ASM-NEXT: vst     %v0, 0(%r3,%r4){{$}}

  vec_xstd2(vd, idx, ptrd);
  // CHECK-ASM-NEXT: vl
  // CHECK-ASM-NEXT: lgfrl   %r3, idx
  // CHECK-ASM-NEXT: lgrl    %r4, ptrd
  // CHECK-ASM-NEXT: vst     %v0, 0(%r3,%r4){{$}}

  vd = vec_splat(vd, 0);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vrepg
  vd = vec_splat(vd, 1);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  // CHECK-ASM: vrepg

  vd = vec_splats(d);
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> zeroinitializer
  // CHECK-ASM: vlrepg

  vd = vec_mergeh(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK-ASM: vmrhg

  vd = vec_mergel(vd, vd);
  // shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <i32 1, i32 3>
  // CHECK-ASM: vmrlg
}

void test_compare(void) {
  // CHECK-ASM-LABEL: test_compare

  vbl = vec_cmpeq(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq", metadata !{{.*}})
  // CHECK-ASM: vfcedb

  vbl = vec_cmpge(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oge", metadata !{{.*}})
  // CHECK-ASM: kdbr
  // CHECK-ASM: kdbr
  // CHECK-ASM: vst

  vbl = vec_cmpgt(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ogt", metadata !{{.*}})
  // CHECK-ASM: kdbr
  // CHECK-ASM: kdbr
  // CHECK-ASM: vst

  vbl = vec_cmple(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !{{.*}})
  // CHECK-ASM: kdbr
  // CHECK-ASM: kdbr
  // CHECK-ASM: vst

  vbl = vec_cmplt(vd, vd);
  // CHECK: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !{{.*}})
  // CHECK-ASM: kdbr
  // CHECK-ASM: kdbr
  // CHECK-ASM: vst

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

  idx = vec_any_eq(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_any_ne(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfcedbs

  idx = vec_any_ge(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

  idx = vec_any_gt(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchdbs

  idx = vec_any_le(vd, vd);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfchedbs

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
  // CHECK: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadb
  vd = vec_msub(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]], metadata !{{.*}})
  // CHECK-ASM: vfmsdb
  vd = vec_sqrt(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.sqrt.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfsqdb

  vd = vec_ld2f(cptrf);
  // CHECK: [[VAL:%[^ ]+]] = load <2 x float>, <2 x float>* %{{.*}}
  // CHECK: call <2 x double> @llvm.experimental.constrained.fpext.v2f64.v2f32(<2 x float> [[VAL]], metadata !{{.*}})
  // (emulated)
  vec_st2f(vd, ptrf);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x float> @llvm.experimental.constrained.fptrunc.v2f32.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK: store <2 x float> [[VAL]], <2 x float>* %{{.*}}
  // (emulated)

  vd = vec_ctd(vsl, 0);
  // CHECK: call <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // (emulated)
  vd = vec_ctd(vul, 0);
  // CHECK: call <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // (emulated)
  vd = vec_ctd(vsl, 1);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK: call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> [[VAL]], <2 x double> <double 5.000000e-01, double 5.000000e-01>, metadata !{{.*}})
  // (emulated)
  vd = vec_ctd(vul, 1);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK: call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> [[VAL]], <2 x double> <double 5.000000e-01, double 5.000000e-01>, metadata !{{.*}})
  // (emulated)
  vd = vec_ctd(vsl, 31);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK: call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> [[VAL]], <2 x double> <double 0x3E00000000000000, double 0x3E00000000000000>, metadata !{{.*}})
  // (emulated)
  vd = vec_ctd(vul, 31);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK: call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> [[VAL]], <2 x double> <double 0x3E00000000000000, double 0x3E00000000000000>, metadata !{{.*}})
  // (emulated)

  vsl = vec_ctsl(vd, 0);
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // (emulated)
  vul = vec_ctul(vd, 0);
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // (emulated)
  vsl = vec_ctsl(vd, 1);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> {{.*}}, <2 x double> <double 2.000000e+00, double 2.000000e+00>, metadata !{{.*}})
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double> [[VAL]], metadata !{{.*}})
  // (emulated)
  vul = vec_ctul(vd, 1);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %{{.*}}, <2 x double> <double 2.000000e+00, double 2.000000e+00>, metadata !{{.*}})
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double> [[VAL]], metadata !{{.*}})
  // (emulated)
  vsl = vec_ctsl(vd, 31);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %{{.*}}, <2 x double> <double 0x41E0000000000000, double 0x41E0000000000000>, metadata !{{.*}})
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double> [[VAL]], metadata !{{.*}})
  // (emulated)
  vul = vec_ctul(vd, 31);
  // CHECK: [[VAL:%[^ ]+]] = tail call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %{{.*}}, <2 x double> <double 0x41E0000000000000, double 0x41E0000000000000>, metadata !{{.*}})
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double> [[VAL]], metadata !{{.*}})
  // (emulated)

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

  vd = vec_roundp(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_ceil(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 6
  vd = vec_roundm(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.floor.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_floor(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.floor.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 7
  vd = vec_roundz(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.trunc.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_trunc(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.trunc.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 5
  vd = vec_roundc(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.nearbyint.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 4, 0
  vd = vec_rint(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfidb %{{.*}}, %{{.*}}, 0, 0
  vd = vec_round(vd);
}
