// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z15 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu z15 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed int vsi;
volatile vector signed long long vsl;
volatile vector unsigned int vui;
volatile vector unsigned long long vul;
volatile vector float vf;
volatile vector double vd;

volatile float f;
volatile double d;

const float * volatile cptrf;
const double * volatile cptrd;

float * volatile ptrf;
double * volatile ptrd;

volatile int idx;

void test_core(void) {
  // CHECK-ASM-LABEL: test_core
  vector float vf2;
  vector double vd2;

  vf += vec_revb(vec_xl(idx, cptrf));
  // CHECK-ASM: vlbrf
  vd += vec_revb(vec_xl(idx, cptrd));
  // CHECK-ASM: vlbrg

  vec_xst(vec_revb(vf), idx, ptrf);
  // CHECK-ASM: vstbrf
  vec_xst(vec_revb(vd), idx, ptrd);
  // CHECK-ASM: vstbrg

  vf += vec_revb(vec_insert_and_zero(cptrf));
  // CHECK-ASM: vllebrzf
  vd += vec_revb(vec_insert_and_zero(cptrd));
  // CHECK-ASM: vllebrzg

  vf += vec_revb(vec_splats(f));
  // CHECK-ASM: vlbrrepf
  vd += vec_revb(vec_splats(d));
  // CHECK-ASM: vlbrrepg

  vf2 = vf;
  vf += vec_revb(vec_insert(f, vec_revb(vf2), 0));
  // CHECK-ASM: vlebrf
  vd2 = vd;
  vd += vec_revb(vec_insert(d, vec_revb(vd2), 0));
  // CHECK-ASM: vlebrg

  f = vec_extract(vec_revb(vf), 0);
  // CHECK-ASM: vstebrf
  d = vec_extract(vec_revb(vd), 0);
  // CHECK-ASM: vstebrg

  vf += vec_reve(vec_xl(idx, cptrf));
  // CHECK-ASM: vlerf
  vd += vec_reve(vec_xl(idx, cptrd));
  // CHECK-ASM: vlerg

  vec_xst(vec_reve(vf), idx, ptrf);
  // CHECK-ASM: vsterf
  vec_xst(vec_reve(vd), idx, ptrd);
  // CHECK-ASM: vsterg
}

void test_float(void) {
  // CHECK-ASM-LABEL: test_float

  vd = vec_double(vsl);
  // CHECK: call <2 x double> @llvm.experimental.constrained.sitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcdgb
  vd = vec_double(vul);
  // CHECK: call <2 x double> @llvm.experimental.constrained.uitofp.v2f64.v2i64(<2 x i64> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcdlgb
  vf = vec_float(vsi);
  // CHECK: call <4 x float> @llvm.experimental.constrained.sitofp.v4f32.v4i32(<4 x i32> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcefb
  vf = vec_float(vui);
  // CHECK: call <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i32(<4 x i32> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcelfb

  vsl = vec_signed(vd);
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptosi.v2i64.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcgdb
  vsi = vec_signed(vf);
  // CHECK: call <4 x i32> @llvm.experimental.constrained.fptosi.v4i32.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vcfeb
  vul = vec_unsigned(vd);
  // CHECK: call <2 x i64> @llvm.experimental.constrained.fptoui.v2i64.v2f64(<2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vclgdb
  vui = vec_unsigned(vf);
  // xHECK: fptoui <4 x float> %{{.*}} to <4 x i32>
  // CHECK: call <4 x i32> @llvm.experimental.constrained.fptoui.v4i32.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vclfeb
}

