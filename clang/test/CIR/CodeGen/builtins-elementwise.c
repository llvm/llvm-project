// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef int vint4 __attribute__((ext_vector_type(4)));
typedef float vfloat4 __attribute__((ext_vector_type(4)));
typedef double vdouble4 __attribute__((ext_vector_type(4)));

void test_builtin_elementwise_acos(float f, double d, vfloat4 vf4,
                                   vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_acos
  // LLVM-LABEL: test_builtin_elementwise_acos
  // OGCG-LABEL: test_builtin_elementwise_acos

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.acos.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.acos.f32(float %{{.*}})
  f = __builtin_elementwise_acos(f);

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.acos.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.acos.f64(double %{{.*}})
  d = __builtin_elementwise_acos(d);

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: %{{.*}} = call <4 x float> @llvm.acos.v4f32(<4 x float> %{{.*}})
  // OGCG: %{{.*}} = call <4 x float> @llvm.acos.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_acos(vf4);

  // CIR: %{{.*}} = cir.acos %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: %{{.*}} = call <4 x double> @llvm.acos.v4f64(<4 x double> %{{.*}})
  // OGCG: %{{.*}} = call <4 x double> @llvm.acos.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_acos(vd4);
}

void test_builtin_elementwise_asin(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_asin
  // LLVM-LABEL: test_builtin_elementwise_asin
  // OGCG-LABEL: test_builtin_elementwise_asin

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.asin.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.asin.f32(float %{{.*}})
  f = __builtin_elementwise_asin(f);

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.asin.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.asin.f64(double %{{.*}})
  d = __builtin_elementwise_asin(d);

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: %{{.*}} = call <4 x float> @llvm.asin.v4f32(<4 x float> %{{.*}})
  // OGCG: %{{.*}} = call <4 x float> @llvm.asin.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_asin(vf4);

  // CIR: %{{.*}} = cir.asin %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: %{{.*}} = call <4 x double> @llvm.asin.v4f64(<4 x double> %{{.*}})
  // OGCG: %{{.*}} = call <4 x double> @llvm.asin.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_asin(vd4);
}

void test_builtin_elementwise_atan(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_atan
  // LLVM-LABEL: test_builtin_elementwise_atan
  // OGCG-LABEL: test_builtin_elementwise_atan

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.float
  // LLVM: %{{.*}} = call float @llvm.atan.f32(float %{{.*}})
  // OGCG: %{{.*}} = call float @llvm.atan.f32(float %{{.*}})
  f = __builtin_elementwise_atan(f);

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.double
  // LLVM: %{{.*}} = call double @llvm.atan.f64(double %{{.*}})
  // OGCG: %{{.*}} = call double @llvm.atan.f64(double %{{.*}})
  d = __builtin_elementwise_atan(d);

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: %{{.*}} = call <4 x float> @llvm.atan.v4f32(<4 x float> %{{.*}})
  // OGCG: %{{.*}} = call <4 x float> @llvm.atan.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_atan(vf4);

  // CIR: %{{.*}} = cir.atan %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: %{{.*}} = call <4 x double> @llvm.atan.v4f64(<4 x double> %{{.*}})
  // OGCG: %{{.*}} = call <4 x double> @llvm.atan.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_atan(vd4);
}
