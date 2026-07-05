// RUN: %clang_cc1 -triple powerpc64le-gnu-linux \
// RUN: -target-feature +vsx -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck      \
// RUN: %s

typedef __attribute__((vector_size(4 * sizeof(float)))) float vec_float;
typedef __attribute__((vector_size(2 * sizeof(double)))) double vec_double;

volatile vec_double vd;
volatile vec_float vf;

void test_fma(void) {
  vf = __builtin_vsx_xvmaddasp(vf, vf, vf);
  // CHECK: @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})

  vd = __builtin_vsx_xvmaddadp(vd, vd, vd);
  // CHECK: @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})

  vf = __builtin_vsx_xvnmaddasp(vf, vf, vf);
  // CHECK: [[RESULT:%[^ ]+]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: fneg <4 x float> [[RESULT]]

  vd = __builtin_vsx_xvnmaddadp(vd, vd, vd);
  // CHECK: [[RESULT:%[^ ]+]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: fneg <2 x double> [[RESULT]]

  vf = __builtin_vsx_xvmsubasp(vf, vf, vf);
  // CHECK: [[RESULT:%[^ ]+]] = fneg <4 x float> %{{.*}}
  // CHECK: @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[RESULT]])

  vd = __builtin_vsx_xvmsubadp(vd, vd, vd);
  // CHECK: [[RESULT:%[^ ]+]] = fneg <2 x double> %{{.*}}
  // CHECK: <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[RESULT]])

  vf = __builtin_vsx_xvnmsubasp(vf, vf, vf);
  // CHECK: call <4 x float> @llvm.ppc.fnmsub.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})

  vd = __builtin_vsx_xvnmsubadp(vd, vd, vd);
  // CHECK: call <2 x double> @llvm.ppc.fnmsub.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
}
