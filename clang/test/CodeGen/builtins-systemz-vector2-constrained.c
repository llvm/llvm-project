// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-ibm-linux -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((vector_size(16))) double vec_double;
typedef __attribute__((vector_size(16))) float vec_float;

volatile vec_double vd;
volatile vec_float vf;

void test_float(void) {
  vd = __builtin_s390_vfmaxdb(vd, vd, 4);
  // CHECK: call <2 x double> @llvm.experimental.constrained.maxnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  vd = __builtin_s390_vfmindb(vd, vd, 4);
  // CHECK: call <2 x double> @llvm.experimental.constrained.minnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  vd = __builtin_s390_vfmindb(vd, vd, 0);

  vd = __builtin_s390_vfnmadb(vd, vd, vd);
  // CHECK: [[RES:%[^ ]+]] = call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: fneg <2 x double> [[RES]]

  vd = __builtin_s390_vfnmsdb(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> {{.*}}
  // CHECK:  [[RES:%[^ ]+]] = call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]], metadata !{{.*}})
  // CHECK: fneg <2 x double> [[RES]]

  vf = __builtin_s390_vfmaxsb(vf, vf, 4);
  // CHECK: call <4 x float> @llvm.experimental.constrained.maxnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})

  vf = __builtin_s390_vfminsb(vf, vf, 4);
  // CHECK: call <4 x float> @llvm.experimental.constrained.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})

  vf = __builtin_s390_vfsqsb(vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.sqrt.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})

  vf = __builtin_s390_vfmasb(vf, vf, vf);
  // CHECK: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  vf = __builtin_s390_vfmssb(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]], metadata !{{.*}})
  vf = __builtin_s390_vfnmasb(vf, vf, vf);
  // CHECK: [[RES:%[^ ]+]] = call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK: fneg <4 x float> [[RES]]
  vf = __builtin_s390_vfnmssb(vf, vf, vf);
  // CHECK: [[NEG:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK: [[RES:%[^ ]+]] = call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]], metadata !{{.*}})
  // CHECK: fneg <4 x float> [[RES]]

  vf = __builtin_s390_vflpsb(vf);
  // CHECK: call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  vf = __builtin_s390_vflnsb(vf);
  // CHECK: [[ABS:%[^ ]+]] = call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
  // CHECK: fneg <4 x float> [[ABS]]

  vf = __builtin_s390_vfisb(vf, 0, 0);
  // CHECK: call <4 x float> @llvm.experimental.constrained.rint.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 0);
  // CHECK: call <4 x float> @llvm.experimental.constrained.nearbyint.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 1);
  // CHECK: call <4 x float> @llvm.experimental.constrained.round.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 5);
  // CHECK: call <4 x float> @llvm.experimental.constrained.trunc.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 6);
  // CHECK: call <4 x float> @llvm.experimental.constrained.ceil.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
  vf = __builtin_s390_vfisb(vf, 4, 7);
  // CHECK: call <4 x float> @llvm.experimental.constrained.floor.v4f32(<4 x float> %{{.*}}, metadata !{{.*}})
}

