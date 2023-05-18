// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-ibm-linux -flax-vector-conversions=none \
// RUN: -ffp-exception-behavior=strict -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((vector_size(16))) signed long long vec_slong;
typedef __attribute__((vector_size(16))) double vec_double;

volatile vec_slong vsl;
volatile vec_double vd;

int cc;

void test_float(void) {
  vsl = __builtin_s390_vfcedbs(vd, vd, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfcedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  vsl = __builtin_s390_vfchdbs(vd, vd, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchdbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  vsl = __builtin_s390_vfchedbs(vd, vd, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vfchedbs(<2 x double> %{{.*}}, <2 x double> %{{.*}})

  vsl = __builtin_s390_vftcidb(vd, 0, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 0)
  vsl = __builtin_s390_vftcidb(vd, 4095, &cc);
  // CHECK: call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %{{.*}}, i32 4095)

  vd = __builtin_s390_vfsqdb(vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.sqrt.v2f64(<2 x double> %{{.*}})

  vd = __builtin_s390_vfmadb(vd, vd, vd);
  // CHECK: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  vd = __builtin_s390_vfmsdb(vd, vd, vd);
  // CHECK: [[NEG:%[^ ]+]] = fneg <2 x double> {{.*}}
  // CHECK: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]], {{.*}})

  vd = __builtin_s390_vflpdb(vd);
  // CHECK: call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vflndb(vd);
  // CHECK: [[ABS:%[^ ]+]] = call <2 x double> @llvm.fabs.v2f64(<2 x double> %{{.*}})
  // CHECK: fneg <2 x double> [[ABS]]

  vd = __builtin_s390_vfidb(vd, 0, 0);
  // CHECK: call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 0);
  // CHECK: call <2 x double> @llvm.experimental.constrained.nearbyint.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 1);
  // CHECK: call <2 x double> @llvm.experimental.constrained.round.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 5);
  // CHECK: call <2 x double> @llvm.experimental.constrained.trunc.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 6);
  // CHECK: call <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 7);
  // CHECK: call <2 x double> @llvm.experimental.constrained.floor.v2f64(<2 x double> %{{.*}})
  vd = __builtin_s390_vfidb(vd, 4, 4);
  // CHECK: call <2 x double> @llvm.s390.vfidb(<2 x double> %{{.*}}, i32 4, i32 4)
}
