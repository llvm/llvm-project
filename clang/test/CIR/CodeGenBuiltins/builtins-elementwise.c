// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:  -emit-llvm  %s -o %t.ll
// RUN: FileCheck  --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-ogcg.ll %s

typedef int vint4 __attribute__((ext_vector_type(4)));
typedef unsigned int vuint4 __attribute__((ext_vector_type(4)));
typedef short vshort8 __attribute__((ext_vector_type(8)));
typedef float vfloat4 __attribute__((ext_vector_type(4)));
typedef double vdouble4 __attribute__((ext_vector_type(4)));

void test_builtin_elementwise_abs(vint4 vi4, int i, float f, double d,
                                  vfloat4 vf4, vdouble4  vd4) {
    // CIR-LABEL: test_builtin_elementwise_abs
    // LLVM-LABEL: test_builtin_elementwise_abs

    // CIR: cir.fabs %{{.*}} : !cir.float
    // LLVM: call float @llvm.fabs.f32(float %{{.*}})
    f = __builtin_elementwise_abs(f);

    // CIR: cir.fabs %{{.*}} : !cir.double
    // LLVM: call double @llvm.fabs.f64(double %{{.*}})
    d = __builtin_elementwise_abs(d);

    // CIR: cir.abs %{{.*}} : !cir.vector<4 x !s32i>
    // LLVM: call <4 x i32> @llvm.abs.v4i32(<4 x i32> %{{.*}}, i1 false)
    vi4 = __builtin_elementwise_abs(vi4);

    // CIR: cir.abs %{{.*}} : !s32
    // LLVM: call i32 @llvm.abs.i32(i32 %{{.*}}, i1 false)
    i = __builtin_elementwise_abs(i);

    // CIR: cir.fabs %{{.*}} : !cir.vector<4 x !cir.float>
    // LLVM: call <4 x float> @llvm.fabs.v4f32(<4 x float> %{{.*}})
    vf4 = __builtin_elementwise_abs(vf4);

    // CIR: cir.fabs %{{.*}} : !cir.vector<4 x !cir.double>
    // LLVM: call <4 x double> @llvm.fabs.v4f64(<4 x double> %{{.*}})
    vd4 = __builtin_elementwise_abs(vd4);
}

void test_builtin_elementwise_acos(float f, double d, vfloat4 vf4,
                                   vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_acos
  // LLVM-LABEL: test_builtin_elementwise_acos
  
  // CIR: cir.acos %{{.*}} : !cir.float
  // LLVM: call float @llvm.acos.f32(float %{{.*}})
  f = __builtin_elementwise_acos(f);

  // CIR: cir.acos %{{.*}} : !cir.double
  // LLVM: call double @llvm.acos.f64(double %{{.*}})
  d = __builtin_elementwise_acos(d);

  // CIR: cir.acos %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.acos.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_acos(vf4);

  // CIR: cir.acos %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.acos.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_acos(vd4);
}

void test_builtin_elementwise_cosh(float f, double d, vfloat4 vf4,
                                   vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_cosh
  // LLVM-LABEL: test_builtin_elementwise_cosh

  // CIR: cir.cosh %{{.*}} : !cir.float
  // LLVM: call float @llvm.cosh.f32(float %{{.*}})
  f = __builtin_elementwise_cosh(f);

  // CIR: cir.cosh %{{.*}} : !cir.double
  // LLVM: call double @llvm.cosh.f64(double %{{.*}})
  d = __builtin_elementwise_cosh(d);

  // CIR: cir.cosh %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.cosh.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_cosh(vf4);

  // CIR: cir.cosh %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.cosh.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_cosh(vd4);
}

void test_builtin_elementwise_asin(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_asin
  // LLVM-LABEL: test_builtin_elementwise_asin

  // CIR: cir.asin %{{.*}} : !cir.float
  // LLVM: call float @llvm.asin.f32(float %{{.*}})
  f = __builtin_elementwise_asin(f);

  // CIR: cir.asin %{{.*}} : !cir.double
  // LLVM: call double @llvm.asin.f64(double %{{.*}})
  d = __builtin_elementwise_asin(d);

  // CIR: cir.asin %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.asin.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_asin(vf4);

  // CIR: cir.asin %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.asin.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_asin(vd4);
}

void test_builtin_elementwise_sinh(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_sinh
  // LLVM-LABEL: test_builtin_elementwise_sinh

  // CIR: cir.sinh %{{.*}} : !cir.float
  // LLVM: call float @llvm.sinh.f32(float %{{.*}})
  f = __builtin_elementwise_sinh(f);

  // CIR: cir.sinh %{{.*}} : !cir.double
  // LLVM: call double @llvm.sinh.f64(double %{{.*}})
  d = __builtin_elementwise_sinh(d);

  // CIR: cir.sinh %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.sinh.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_sinh(vf4);

  // CIR: cir.sinh %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.sinh.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_sinh(vd4);
}

void test_builtin_elementwise_atan(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_atan
  // LLVM-LABEL: test_builtin_elementwise_atan

  // CIR: cir.atan %{{.*}} : !cir.float
  // LLVM: call float @llvm.atan.f32(float %{{.*}})
  f = __builtin_elementwise_atan(f);

  // CIR: cir.atan %{{.*}} : !cir.double
  // LLVM: call double @llvm.atan.f64(double %{{.*}})
  d = __builtin_elementwise_atan(d);

  // CIR: cir.atan %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.atan.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_atan(vf4);

  // CIR: cir.atan %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.atan.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_atan(vd4);
}

void test_builtin_elementwise_tanh(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_tanh
  // LLVM-LABEL: test_builtin_elementwise_tanh

  // CIR: cir.tanh %{{.*}} : !cir.float
  // LLVM: call float @llvm.tanh.f32(float %{{.*}})
  f = __builtin_elementwise_tanh(f);

  // CIR: cir.tanh %{{.*}} : !cir.double
  // LLVM: call double @llvm.tanh.f64(double %{{.*}})
  d = __builtin_elementwise_tanh(d);

  // CIR: cir.tanh %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.tanh.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_tanh(vf4);

  // CIR: cir.tanh %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.tanh.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_tanh(vd4);
}

void test_builtin_elementwise_atan2(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_atan2
  // LLVM-LABEL: test_builtin_elementwise_atan2

  // CIR: cir.atan2 %{{.*}}, %{{.*}} : !cir.float
  // LLVM: call float @llvm.atan2.f32(float %{{.*}}, float %{{.*}})
  f = __builtin_elementwise_atan2(f, f);

  // CIR: cir.atan2 %{{.*}}, %{{.*}} : !cir.double
  // LLVM: call double @llvm.atan2.f64(double %{{.*}}, double %{{.*}})
  d = __builtin_elementwise_atan2(d, d);

  // CIR: cir.atan2 %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.atan2.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  vf4 = __builtin_elementwise_atan2(vf4, vf4);

  // CIR: cir.atan2 %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.atan2.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}})
  vd4 = __builtin_elementwise_atan2(vd4, vd4);
}

void test_builtin_elementwise_exp(float f, double d, vfloat4 vf4,
                                  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_exp
  // LLVM-LABEL: test_builtin_elementwise_exp

  // CIR: cir.exp %{{.*}} : !cir.float
  // LLVM: call float @llvm.exp.f32(float %{{.*}})
  f = __builtin_elementwise_exp(f);

  // CIR: cir.exp %{{.*}} : !cir.double
  // LLVM: call double @llvm.exp.f64(double %{{.*}})
  d = __builtin_elementwise_exp(d);

  // CIR: cir.exp %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.exp.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_exp(vf4);

  // CIR: cir.exp %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.exp.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_exp(vd4);
}

void test_builtin_elementwise_exp2(float f, double d, vfloat4 vf4,
                                  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_exp2
  // LLVM-LABEL: test_builtin_elementwise_exp2

  // CIR: cir.exp2 %{{.*}} : !cir.float
  // LLVM: call float @llvm.exp2.f32(float %{{.*}})
  f = __builtin_elementwise_exp2(f);

  // CIR: cir.exp2 %{{.*}} : !cir.double
  // LLVM: call double @llvm.exp2.f64(double %{{.*}})
  d = __builtin_elementwise_exp2(d);

  // CIR: cir.exp2 %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.exp2.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_exp2(vf4);

  // CIR: cir.exp2 %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.exp2.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_exp2(vd4);
}

void test_builtin_elementwise_exp10(float f, double d, vfloat4 vf4,
                                  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_exp10
  // LLVM-LABEL: test_builtin_elementwise_exp10

  // CIR: cir.exp10 %{{.*}} : !cir.float
  // LLVM: call float @llvm.exp10.f32(float %{{.*}})
  f = __builtin_elementwise_exp10(f);

  // CIR: cir.exp10 %{{.*}} : !cir.double
  // LLVM: call double @llvm.exp10.f64(double %{{.*}})
  d = __builtin_elementwise_exp10(d);

  // CIR: cir.exp10 %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.exp10.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_exp10(vf4);

  // CIR: cir.exp10 %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.exp10.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_exp10(vd4);
}

void test_builtin_elementwise_log(float f, double d, vfloat4 vf4,
                                  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_log
  // LLVM-LABEL: test_builtin_elementwise_log

  // CIR: cir.log %{{.*}} : !cir.float
  // LLVM: call float @llvm.log.f32(float %{{.*}})
  f = __builtin_elementwise_log(f);

  // CIR: cir.log %{{.*}} : !cir.double
  // LLVM: call double @llvm.log.f64(double %{{.*}})
  d = __builtin_elementwise_log(d);

  // CIR: cir.log %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.log.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_log(vf4);

  // CIR: cir.log %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.log.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_log(vd4);
}

void test_builtin_elementwise_log2(float f, double d, vfloat4 vf4,
                                    vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_log2
  // LLVM-LABEL: test_builtin_elementwise_log2

  // CIR: cir.log2 %{{.*}} : !cir.float
  // LLVM: call float @llvm.log2.f32(float %{{.*}})
  f = __builtin_elementwise_log2(f);

  // CIR: cir.log2 %{{.*}} : !cir.double
  // LLVM: call double @llvm.log2.f64(double %{{.*}})
  d = __builtin_elementwise_log2(d);

  // CIR: cir.log2 %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.log2.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_log2(vf4);

  // CIR: cir.log2 %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.log2.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_log2(vd4);
}

void test_builtin_elementwise_log10(float f, double d, vfloat4 vf4,
                                     vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_log10
  // LLVM-LABEL: test_builtin_elementwise_log10

  // CIR: cir.log10 %{{.*}} : !cir.float
  // LLVM: call float @llvm.log10.f32(float %{{.*}})
  f = __builtin_elementwise_log10(f);

  // CIR: cir.log10 %{{.*}} : !cir.double
  // LLVM: call double @llvm.log10.f64(double %{{.*}})
  d = __builtin_elementwise_log10(d);

  // CIR: cir.log10 %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.log10.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_log10(vf4);

  // CIR: cir.log10 %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.log10.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_log10(vd4);
}

void test_builtin_elementwise_cos(float f, double d, vfloat4 vf4,
                                     vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_cos
  // LLVM-LABEL: test_builtin_elementwise_cos

  // CIR: cir.cos %{{.*}} : !cir.float
  // LLVM: call float @llvm.cos.f32(float %{{.*}})
  f = __builtin_elementwise_cos(f);

  // CIR: cir.cos %{{.*}} : !cir.double
  // LLVM: call double @llvm.cos.f64(double %{{.*}})
  d = __builtin_elementwise_cos(d);

  // CIR: cir.cos %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.cos.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_cos(vf4);

  // CIR: cir.cos %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.cos.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_cos(vd4);
}

void test_builtin_elementwise_ceil(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_ceil
  // LLVM-LABEL: test_builtin_elementwise_ceil

  // CIR: cir.ceil %{{.*}} : !cir.float
  // LLVM: call float @llvm.ceil.f32(float %{{.*}})
  f = __builtin_elementwise_ceil(f);

  // CIR: cir.ceil %{{.*}} : !cir.double
  // LLVM: call double @llvm.ceil.f64(double %{{.*}})
  d = __builtin_elementwise_ceil(d);

  // CIR: cir.ceil %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.ceil.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_ceil(vf4);

  // CIR: cir.ceil %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.ceil.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_ceil(vd4);
}

void test_builtin_elementwise_floor(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_floor
  // LLVM-LABEL: test_builtin_elementwise_floor

  // CIR: cir.floor %{{.*}} : !cir.float
  // LLVM: call float @llvm.floor.f32(float %{{.*}})
  f = __builtin_elementwise_floor(f);

  // CIR: cir.floor %{{.*}} : !cir.double
  // LLVM: call double @llvm.floor.f64(double %{{.*}})
  d = __builtin_elementwise_floor(d);

  // CIR: cir.floor %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.floor.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_floor(vf4);

  // CIR: cir.floor %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.floor.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_floor(vd4);
}

void test_builtin_elementwise_fmod(float f, double d, vfloat4 vf4,
                                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_fmod
  // LLVM-LABEL: test_builtin_elementwise_fmod

  // CIR: cir.fmod %{{.*}}, %{{.*}} : !cir.float
  // LLVM: frem float %{{.*}}, %{{.*}}
  f = __builtin_elementwise_fmod(f, f);

  // CIR: cir.fmod %{{.*}}, %{{.*}} : !cir.double
  // LLVM: frem double %{{.*}}, %{{.*}}
  d = __builtin_elementwise_fmod(d, d);

  // CIR: cir.fmod %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: frem <4 x float> %{{.*}}, %{{.*}}
  vf4 = __builtin_elementwise_fmod(vf4, vf4);

  // CIR: cir.fmod %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: frem <4 x double> %{{.*}}, %{{.*}}
  vd4 = __builtin_elementwise_fmod(vd4, vd4);
}

void test_builtin_elementwise_roundeven(float f, double d, vfloat4 vf4,
                                        vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_roundeven
  // LLVM-LABEL: test_builtin_elementwise_roundeven

  // CIR: cir.roundeven %{{.*}} : !cir.float
  // LLVM: call float @llvm.roundeven.f32(float %{{.*}})
  f = __builtin_elementwise_roundeven(f);

  // CIR: cir.roundeven %{{.*}} : !cir.double
  // LLVM: call double @llvm.roundeven.f64(double %{{.*}})
  d = __builtin_elementwise_roundeven(d);

  // CIR: cir.roundeven %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.roundeven.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_roundeven(vf4);

  // CIR: cir.roundeven %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.roundeven.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_roundeven(vd4);
}

void test_builtin_elementwise_round(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_round
  // LLVM-LABEL: test_builtin_elementwise_round

  // CIR: cir.round %{{.*}} : !cir.float
  // LLVM: call float @llvm.round.f32(float %{{.*}})
  f = __builtin_elementwise_round(f);

  // CIR: cir.round %{{.*}} : !cir.double
  // LLVM: call double @llvm.round.f64(double %{{.*}})
  d = __builtin_elementwise_round(d);

  // CIR: cir.round %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.round.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_round(vf4);

  // CIR: cir.round %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.round.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_round(vd4);
}

void test_builtin_elementwise_rint(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_rint
  // LLVM-LABEL: test_builtin_elementwise_rint

  // CIR: cir.rint %{{.*}} : !cir.float
  // LLVM: call float @llvm.rint.f32(float %{{.*}})
  f = __builtin_elementwise_rint(f);

  // CIR: cir.rint %{{.*}} : !cir.double
  // LLVM: call double @llvm.rint.f64(double %{{.*}})
  d = __builtin_elementwise_rint(d);

  // CIR: cir.rint %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.rint.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_rint(vf4);

  // CIR: cir.rint %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.rint.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_rint(vd4);
}

void test_builtin_elementwise_nearbyint(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_nearbyint
  // LLVM-LABEL: test_builtin_elementwise_nearbyint

  // CIR: cir.nearbyint %{{.*}} : !cir.float
  // LLVM: call float @llvm.nearbyint.f32(float %{{.*}})
  f = __builtin_elementwise_nearbyint(f);

  // CIR: cir.nearbyint %{{.*}} : !cir.double
  // LLVM: call double @llvm.nearbyint.f64(double %{{.*}})
  d = __builtin_elementwise_nearbyint(d);

  // CIR: cir.nearbyint %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_nearbyint(vf4);

  // CIR: cir.nearbyint %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_nearbyint(vd4);
}

void test_builtin_elementwise_sin(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_sin
  // LLVM-LABEL: test_builtin_elementwise_sin

  // CIR: cir.sin %{{.*}} : !cir.float
  // LLVM: call float @llvm.sin.f32(float %{{.*}})
  f = __builtin_elementwise_sin(f);

  // CIR: cir.sin %{{.*}} : !cir.double
  // LLVM: call double @llvm.sin.f64(double %{{.*}})
  d = __builtin_elementwise_sin(d);

  // CIR: cir.sin %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.sin.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_sin(vf4);

  // CIR: cir.sin %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.sin.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_sin(vd4);
}

void test_builtin_elementwise_sqrt(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_sqrt
  // LLVM-LABEL: test_builtin_elementwise_sqrt

  // CIR: cir.sqrt %{{.*}} : !cir.float
  // LLVM: call float @llvm.sqrt.f32(float %{{.*}})
  f = __builtin_elementwise_sqrt(f);

  // CIR: cir.sqrt %{{.*}} : !cir.double
  // LLVM: call double @llvm.sqrt.f64(double %{{.*}})
  d = __builtin_elementwise_sqrt(d);

  // CIR: cir.sqrt %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_sqrt(vf4);

  // CIR: cir.sqrt %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.sqrt.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_sqrt(vd4);
}

void test_builtin_elementwise_tan(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_tan
  // LLVM-LABEL: test_builtin_elementwise_tan

  // CIR: cir.tan %{{.*}} : !cir.float
  // LLVM: call float @llvm.tan.f32(float %{{.*}})
  f = __builtin_elementwise_tan(f);

  // CIR: cir.tan %{{.*}} : !cir.double
  // LLVM: call double @llvm.tan.f64(double %{{.*}})
  d = __builtin_elementwise_tan(d);

  // CIR: cir.tan %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.tan.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_tan(vf4);

  // CIR: cir.tan %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.tan.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_tan(vd4);
}

void test_builtin_elementwise_trunc(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_trunc
  // LLVM-LABEL: test_builtin_elementwise_trunc
  
  // CIR: cir.trunc %{{.*}} : !cir.float
  // LLVM: call float @llvm.trunc.f32(float %{{.*}})
  f = __builtin_elementwise_trunc(f);

  // CIR: cir.trunc %{{.*}} : !cir.double
  // LLVM: call double @llvm.trunc.f64(double %{{.*}})
  d = __builtin_elementwise_trunc(d);

  // CIR: cir.trunc %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.trunc.v4f32(<4 x float> %{{.*}})
  vf4 = __builtin_elementwise_trunc(vf4);

  // CIR: cir.trunc %{{.*}} : !cir.vector<4 x !cir.double>
  // LLVM: call <4 x double> @llvm.trunc.v4f64(<4 x double> %{{.*}})
  vd4 = __builtin_elementwise_trunc(vd4);
}


void test_builtin_elementwise_fshl(long long int i1, long long int i2,
                                   long long int i3, unsigned short us1,
                                   unsigned short us2, unsigned short us3,
                                   char c1, char c2, char c3,
                                   unsigned char uc1, unsigned char uc2,
                                   unsigned char uc3, vshort8 vi1,
                                   vshort8 vi2, vshort8 vi3, vint4 vu1,
                                   vint4 vu2, vint4 vu3) {
  // CIR-LABEL: test_builtin_elementwise_fshl
  // LLVM-LABEL: test_builtin_elementwise_fshl

  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!s64i, !s64i, !s64i) -> !s64i
  // LLVM: call i64 @llvm.fshl.i64(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  i1 = __builtin_elementwise_fshl(i1, i2, i3);

  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!u16i, !u16i, !u16i) -> !u16i
  // LLVM: call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  us1 = __builtin_elementwise_fshl(us1, us2, us3);

  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!s8i, !s8i, !s8i) -> !s8i
  // LLVM: call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  c1 = __builtin_elementwise_fshl(c1, c2, c3);

  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!u8i, !u8i, !u8i) -> !u8i
  // LLVM: call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  uc1 = __builtin_elementwise_fshl(uc1, uc2, uc3);

  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>
  // LLVM: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vi1 = __builtin_elementwise_fshl(vi1, vi2, vi3);

  // CIR: cir.call_llvm_intrinsic "fshl" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>
  // LLVM: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vu1 = __builtin_elementwise_fshl(vu1, vu2, vu3);
}

void test_builtin_elementwise_fshr(long long int i1, long long int i2,
                                   long long int i3, unsigned short us1,
                                   unsigned short us2, unsigned short us3,
                                   char c1, char c2, char c3,
                                   unsigned char uc1, unsigned char uc2,
                                   unsigned char uc3, vshort8 vi1,
                                   vshort8 vi2, vshort8 vi3, vint4 vu1,
                                   vint4 vu2, vint4 vu3) {
  // CIR-LABEL: test_builtin_elementwise_fshr
  // LLVM-LABEL: test_builtin_elementwise_fshr

  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!s64i, !s64i, !s64i) -> !s64i
  // LLVM: call i64 @llvm.fshr.i64(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  i1 = __builtin_elementwise_fshr(i1, i2, i3);

  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!u16i, !u16i, !u16i) -> !u16i
  // LLVM: call i16 @llvm.fshr.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  us1 = __builtin_elementwise_fshr(us1, us2, us3);

  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!s8i, !s8i, !s8i) -> !s8i
  // LLVM: call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  c1 = __builtin_elementwise_fshr(c1, c2, c3);

  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!u8i, !u8i, !u8i) -> !u8i
  // LLVM: call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 %{{.*}})
  uc1 = __builtin_elementwise_fshr(uc1, uc2, uc3);

  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>
  // LLVM: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vi1 = __builtin_elementwise_fshr(vi1, vi2, vi3);

  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>
  // LLVM: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vu1 = __builtin_elementwise_fshr(vu1, vu2, vu3);
}

void test_builtin_elementwise_add_sat(int i1, int i2, unsigned u1, unsigned u2,
                                      short s1, short s2, vint4 vi1, vint4 vi2,
                                      vuint4 vu1, vuint4 vu2, vshort8 vs1,
                                      vshort8 vs2) {
  // CIR-LABEL: test_builtin_elementwise_add_sat
  // LLVM-LABEL: test_builtin_elementwise_add_sat

  // CIR: cir.add sat %{{.*}}, %{{.*}} : !s32i
  // LLVM: call i32 @llvm.sadd.sat.i32(i32 %{{.*}}, i32 %{{.*}})
  i1 = __builtin_elementwise_add_sat(i1, i2);

  // CIR: cir.add sat %{{.*}}, %{{.*}} : !u32i
  // LLVM: call i32 @llvm.uadd.sat.i32(i32 %{{.*}}, i32 %{{.*}})
  u1 = __builtin_elementwise_add_sat(u1, u2);

  // CIR: cir.add sat %{{.*}}, %{{.*}} : !s16i
  // LLVM: call i16 @llvm.sadd.sat.i16(i16 %{{.*}}, i16 %{{.*}})
  s1 = __builtin_elementwise_add_sat(s1, s2);

  // CIR: cir.add sat %{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>
  // LLVM: call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vi1 = __builtin_elementwise_add_sat(vi1, vi2);

  // CIR: cir.add sat %{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>
  // LLVM: call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vu1 = __builtin_elementwise_add_sat(vu1, vu2);

  // CIR: cir.add sat %{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>
  // LLVM: call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vs1 = __builtin_elementwise_add_sat(vs1, vs2);
}

void test_builtin_elementwise_sub_sat(int i1, int i2, unsigned u1, unsigned u2,
                                      short s1, short s2, vint4 vi1, vint4 vi2,
                                      vuint4 vu1, vuint4 vu2, vshort8 vs1,
                                      vshort8 vs2) {
  // CIR-LABEL: test_builtin_elementwise_sub_sat
  // LLVM-LABEL: test_builtin_elementwise_sub_sat

  // CIR: cir.sub sat %{{.*}}, %{{.*}} : !s32i
  // LLVM: call i32 @llvm.ssub.sat.i32(i32 %{{.*}}, i32 %{{.*}})
  i1 = __builtin_elementwise_sub_sat(i1, i2);

  // CIR: cir.sub sat %{{.*}}, %{{.*}} : !u32i
  // LLVM: call i32 @llvm.usub.sat.i32(i32 %{{.*}}, i32 %{{.*}})
  u1 = __builtin_elementwise_sub_sat(u1, u2);

  // CIR: cir.sub sat %{{.*}}, %{{.*}} : !s16i
  // LLVM: call i16 @llvm.ssub.sat.i16(i16 %{{.*}}, i16 %{{.*}})
  s1 = __builtin_elementwise_sub_sat(s1, s2);

  // CIR: cir.sub sat %{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>
  // LLVM: call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vi1 = __builtin_elementwise_sub_sat(vi1, vi2);

  // CIR: cir.sub sat %{{.*}}, %{{.*}} : !cir.vector<4 x !u32i>
  // LLVM: call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  vu1 = __builtin_elementwise_sub_sat(vu1, vu2);

  // CIR: cir.sub sat %{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>
  // LLVM: call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  vs1 = __builtin_elementwise_sub_sat(vs1, vs2);
}

vfloat4 test_builtin_elementwise_fma(vfloat4 a, vfloat4 b, vfloat4 c) {
  // CIR-LABEL: test_builtin_elementwise_fma
  // LLVM-LABEL: test_builtin_elementwise_fma

  // CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>
  // LLVM: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return __builtin_elementwise_fma(a, b, c);
}

typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef short int si8 __attribute__((ext_vector_type(8)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef unsigned int u4 __attribute__((ext_vector_type(4)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double3 __attribute__((ext_vector_type(3)));
__attribute__((address_space(1))) int int_as_one;
typedef int bar;
bar b;

void test_builtin_elementwise_min(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2, long long int i1,
                                  long long int i2, si8 vi1, si8 vi2,
                                  unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                  _BitInt(31) bi1, _BitInt(31) bi2,
                                  unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CIR-LABEL: @test_builtin_elementwise_min
  // LLVM-LABEL: @test_builtin_elementwise_min(

  // CIR: %[[F1:.*]] = cir.alloca "f1" align(4) init : !cir.ptr<!cir.float>
  // CIR: %[[F2:.*]] = cir.alloca "f2" align(4) init : !cir.ptr<!cir.float>
  // CIR: %[[D1:.*]] = cir.alloca "d1" align(8) init : !cir.ptr<!cir.double>
  // CIR: %[[D2:.*]] = cir.alloca "d2" align(8) init : !cir.ptr<!cir.double>
  // CIR: %[[VF1:.*]] = cir.alloca "vf1" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR: %[[VF2:.*]] = cir.alloca "vf2" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR: %[[I1:.*]] = cir.alloca "i1" align(8) init : !cir.ptr<!s64i>
  // CIR: %[[I2:.*]] = cir.alloca "i2" align(8) init : !cir.ptr<!s64i>
  // CIR: %[[VI1:.*]] = cir.alloca "vi1" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR: %[[VI2:.*]] = cir.alloca "vi2" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR: %[[U1:.*]] = cir.alloca "u1" align(4) init : !cir.ptr<!u32i>
  // CIR: %[[U2:.*]] = cir.alloca "u2" align(4) init : !cir.ptr<!u32i>
  // CIR: %[[VU1:.*]] = cir.alloca "vu1" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR: %[[VU2:.*]] = cir.alloca "vu2" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR: %[[BI1:.*]] = cir.alloca "bi1" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR: %[[BI2:.*]] = cir.alloca "bi2" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR: %[[BU1:.*]] = cir.alloca "bu1" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR: %[[BU2:.*]] = cir.alloca "bu2" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR: %[[CVF1:.*]] = cir.alloca "cvf1" align(16) init const : !cir.ptr<!cir.vector<4 x !cir.float>>

  // LLVM: %[[ADDR_F1:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_F2:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_D1:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_D2:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_VF1:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_VF2:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_I1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_I2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_VI1:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_VI2:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_U1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_U2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_VU1:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_VU2:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_BI1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BI2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BU1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_BU2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_CVF1:.*]] = alloca <4 x float>, align 16

  // CIR:      %[[F1_LOAD:.*]] = cir.load align(4) %[[F1]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: %[[F2_LOAD:.*]] = cir.load align(4) %[[F2]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: cir.call_llvm_intrinsic "minnum" %[[F1_LOAD]], %[[F2_LOAD]] : (!cir.float, !cir.float) -> !cir.float

  // LLVM:      [[F1:%.+]] = load float, ptr %[[ADDR_F1]], align 4
  // LLVM-NEXT: [[F2:%.+]] = load float, ptr %[[ADDR_F2]], align 4
  // LLVM-NEXT:  call float @llvm.minnum.f32(float [[F1]], float [[F2]])
  f1 = __builtin_elementwise_min(f1, f2);

  // CIR:      %[[D1_LOAD:.*]] = cir.load align(8) %[[D1]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: %[[D2_LOAD:.*]] = cir.load align(8) %[[D2]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "minnum" %[[D1_LOAD]], %[[D2_LOAD]] : (!cir.double, !cir.double) -> !cir.double

  // LLVM:      [[D1:%.+]] = load double, ptr %[[ADDR_D1]], align 8
  // LLVM-NEXT: [[D2:%.+]] = load double, ptr %[[ADDR_D2]], align 8
  // LLVM-NEXT: call double @llvm.minnum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_min(d1, d2);

  // CIR:      %[[D1_LOAD:.*]] = cir.load align(8) %[[D1]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: %[[TWO:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "minnum" %[[D1_LOAD]], %[[TWO]] : (!cir.double, !cir.double) -> !cir.double
 
  // LLVM:      [[D1:%.+]] = load double, ptr %[[ADDR_D1]], align 8
  // LLVM-NEXT: call double @llvm.minnum.f64(double [[D1]], double 2.000000e+00)
  d1 = __builtin_elementwise_min(d1, 2.0);

  // CIR:      %[[VF1_LOAD:.*]] = cir.load align(16) %[[VF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "minnum" %[[VF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF1:%.+]] = load <4 x float>, ptr %[[ADDR_VF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_min(vf1, vf2);

  // CIR:      %[[I1_LOAD:.*]] = cir.load align(8) %[[I1]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: %[[I2_LOAD:.*]] = cir.load align(8) %[[I2]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: cir.call_llvm_intrinsic "smin" %31, %32 : (!s64i, !s64i) -> !s64i

  // LLVM:      [[I1:%.+]] = load i64, ptr %[[ADDR_I1]], align 8
  // LLVM-NEXT: [[I2:%.+]] = load i64, ptr %[[ADDR_I2]], align 8
  // LLVM-NEXT: call i64 @llvm.smin.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_min(i1, i2);

  // CIR:      %[[NEG_11:.*]] = cir.const #cir.int<-11> : !s64i
  // CIR-NEXT: %[[I2_LOAD:.*]] = cir.load align(8) %[[I2]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: cir.call_llvm_intrinsic "smin" %[[NEG_11]], %[[I2_LOAD]] : (!s64i, !s64i) -> !s64i

  // LLVM:      [[I2:%.+]] = load i64, ptr %[[ADDR_I2]], align 8
  // LLVM-NEXT: call i64 @llvm.smin.i64(i64 -11, i64 [[I2]])
  i1 = __builtin_elementwise_min(-11ll, i2);

  // CIR:      %[[I1_LOAD:.*]] = cir.load align(8) %[[I1]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: %[[I1_TRUNC:.*]] = cir.cast integral %[[I1_LOAD]] : !s64i -> !s16i
  // CIR-NEXT: %[[I2_LOAD:.*]] = cir.load align(8) %[[I2]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: %[[I2_TRUNC:.*]] = cir.cast integral %[[I2_LOAD]] : !s64i -> !s16i
  // CIR-NEXT: cir.call_llvm_intrinsic "smin" %[[I1_TRUNC]], %[[I2_TRUNC]] : (!s16i, !s16i) -> !s16i

  // LLVM:      [[I1:%.+]] = load i64, ptr %[[ADDR_I1]], align 8
  // LLVM:      [[S1:%.+]] = trunc i64 [[I1]] to i16
  // LLVM-NEXT: [[I2:%.+]] = load i64, ptr %[[ADDR_I2]], align 8
  // LLVM:      [[S2:%.+]] = trunc i64 [[I2]] to i16
  // LLVM-NEXT: call i16 @llvm.smin.i16(i16 [[S1]], i16 [[S2]])
  i1 = __builtin_elementwise_min((short)i1, (short)i2);

  // CIR:      %[[VI1_LOAD:.*]] = cir.load align(16) %[[VI1]] : !cir.ptr<!cir.vector<8 x !s16i>>, !cir.vector<8 x !s16i>
  // CIR-NEXT: %[[VI2_LOAD:.*]] = cir.load align(16) %[[VI2]] : !cir.ptr<!cir.vector<8 x !s16i>>, !cir.vector<8 x !s16i>
  // CIR-NEXT: cir.call_llvm_intrinsic "smin" %[[VI1_LOAD]], %[[VI2_LOAD]] : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>
  // LLVM:      [[VI1:%.+]] = load <8 x i16>, ptr %[[ADDR_VI1]], align 16
  // LLVM-NEXT: [[VI2:%.+]] = load <8 x i16>, ptr %[[ADDR_VI2]], align 16
  // LLVM-NEXT: call <8 x i16> @llvm.smin.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_min(vi1, vi2);

  // CIR:      %[[U1_LOAD:.*]] = cir.load align(4) %[[U1]] : !cir.ptr<!u32i>, !u32i
  // CIR-NEXT: %[[U2_LOAD:.*]] = cir.load align(4) %[[U2]] : !cir.ptr<!u32i>, !u32i
  // CIR-NEXT: cir.call_llvm_intrinsic "umin" %[[U1_LOAD]], %[[U2_LOAD]] : (!u32i, !u32i) -> !u32i

  // LLVM:      [[U1:%.+]] = load i32, ptr %[[ADDR_U1]], align 4
  // LLVM-NEXT: [[U2:%.+]] = load i32, ptr %[[ADDR_U2]], align 4
  // LLVM-NEXT: call i32 @llvm.umin.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_min(u1, u2);

  // CIR:      %[[VU1_LOAD:.*]] = cir.load align(16) %[[VU1]] : !cir.ptr<!cir.vector<4 x !u32i>>, !cir.vector<4 x !u32i>
  // CIR-NEXT: %[[VU2_LOAD:.*]] = cir.load align(16) %[[VU2]] : !cir.ptr<!cir.vector<4 x !u32i>>, !cir.vector<4 x !u32i>
  // CIR-NEXT: cir.call_llvm_intrinsic "umin" %[[VU1_LOAD]], %[[VU2_LOAD]] : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>) -> !cir.vector<4 x !u32i>

  // LLVM:      [[VU1:%.+]] = load <4 x i32>, ptr %[[ADDR_VU1]], align 16
  // LLVM-NEXT: [[VU2:%.+]] = load <4 x i32>, ptr %[[ADDR_VU2]], align 16
  // LLVM-NEXT: call <4 x i32> @llvm.umin.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_min(vu1, vu2);

  // CIR:      %[[BI1_LOAD:.*]] = cir.load align(4) %[[BI1]] : !cir.ptr<!cir.int<s, 31, bitint>>, !cir.int<s, 31, bitint>
  // CIR-NEXT: %[[BI2_LOAD:.*]] = cir.load align(4) %[[BI2]] : !cir.ptr<!cir.int<s, 31, bitint>>, !cir.int<s, 31, bitint>
  // CIR-NEXT: cir.call_llvm_intrinsic "smin" %[[BI1_LOAD]], %[[BI2_LOAD]] : (!cir.int<s, 31, bitint>, !cir.int<s, 31, bitint>) -> !cir.int<s, 31, bitint>

  // LLVM:      [[BI1:%.+]] = load i32, ptr %[[ADDR_BI1]], align 4
  // LLVM-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // LLVM-NEXT: [[BI2:%.+]] = load i32, ptr %[[ADDR_BI2]], align 4
  // LLVM-NEXT: [[LOADEDV1:%.+]] = trunc i32 [[BI2]] to i31
  // LLVM-NEXT: call i31 @llvm.smin.i31(i31 [[LOADEDV]], i31 [[LOADEDV1]])
  bi1 = __builtin_elementwise_min(bi1, bi2);

  // CIR:      %[[BU1_LOAD:.*]] = cir.load align(8) %[[BU1]] : !cir.ptr<!cir.int<u, 55, bitint>>, !cir.int<u, 55, bitint>
  // CIR-NEXT: %[[BU2_LOAD:.*]] = cir.load align(8) %[[BU2]] : !cir.ptr<!cir.int<u, 55, bitint>>, !cir.int<u, 55, bitint>
  // CIR-NEXT: cir.call_llvm_intrinsic "umin" %[[BU1_LOAD]], %[[BU2_LOAD]] : (!cir.int<u, 55, bitint>, !cir.int<u, 55, bitint>) -> !cir.int<u, 55, bitint>

  // LLVM:      [[BU1:%.+]] = load i64, ptr %[[ADDR_BU1]], align 8
  // LLVM-NEXT: [[LOADEDV2:%.+]] = trunc i64 [[BU1]] to i55
  // LLVM-NEXT: [[BU2:%.+]] = load i64, ptr %[[ADDR_BU2]], align 8
  // LLVM-NEXT: [[LOADEDV3:%.+]] = trunc i64 [[BU2]] to i55
  // LLVM-NEXT: call i55 @llvm.umin.i55(i55 [[LOADEDV2]], i55 [[LOADEDV3]])
  bu1 = __builtin_elementwise_min(bu1, bu2);

  // CIR:      %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "minnum" %[[CVF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_min(cvf1, vf2);

  // CIR:      %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "minnum" %[[VF2_LOAD]], %[[CVF1_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.minnum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_min(vf2, cvf1);

  // CIR:      %[[IAONE:.*]] = cir.get_global @int_as_one : !cir.ptr<!s32i, target_address_space(1)>
  // CIR-NEXT: %[[IAO_LOAD:.*]] = cir.load align(4) %[[IAONE]] : !cir.ptr<!s32i, target_address_space(1)>, !s32i
  // CIR-NEXT: %[[B:.*]] = cir.get_global @b : !cir.ptr<!s32i>
  // CIR-NEXT: %[[B_LOAD:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT: cir.call_llvm_intrinsic "smin" %[[IAO_LOAD]], %[[B_LOAD]] : (!s32i, !s32i) -> !s32i

  // LLVM:      [[IAS1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // LLVM-NEXT: [[B:%.+]] = load i32, ptr @b, align 4
  // LLVM-NEXT: call i32 @llvm.smin.i32(i32 [[IAS1]], i32 [[B]])
  int_as_one = __builtin_elementwise_min(int_as_one, b);

  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
  // CIR-NEXT: cir.store align(8) %[[TWO]], %[[I1]] : !s64i, !cir.ptr<!s64i>

  // LLVM: store i64 2, ptr [[I1:%.+]], align 8
  i1 = __builtin_elementwise_min(2, 'b');
}

void test_builtin_elementwise_minimum(float f1, float f2, double d1, double d2,
                                      float4 vf1, float4 vf2, long long int i1,
                                      long long int i2, si8 vi1, si8 vi2,
                                      unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                      _BitInt(31) bi1, _BitInt(31) bi2,
                                      unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CIR-LABEL: @test_builtin_elementwise_minimum(
  // LLVM-LABEL: @test_builtin_elementwise_minimum(

  // CIR-NEXT: %[[F1:.*]] = cir.alloca "f1" align(4) init : !cir.ptr<!cir.float>
  // CIR-NEXT: %[[F2:.*]] = cir.alloca "f2" align(4) init : !cir.ptr<!cir.float>
  // CIR-NEXT: %[[D1:.*]] = cir.alloca "d1" align(8) init : !cir.ptr<!cir.double>
  // CIR-NEXT: %[[D2:.*]] = cir.alloca "d2" align(8) init : !cir.ptr<!cir.double>
  // CIR-NEXT: %[[VF1:.*]] = cir.alloca "vf1" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR-NEXT: %[[VF2:.*]] = cir.alloca "vf2" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR-NEXT: %[[I1:.*]] = cir.alloca "i1" align(8) init : !cir.ptr<!s64i>
  // CIR-NEXT: %[[I1:.*]] = cir.alloca "i2" align(8) init : !cir.ptr<!s64i>
  // CIR-NEXT: %[[VI1:.*]] = cir.alloca "vi1" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR-NEXT: %[[VI2:.*]] = cir.alloca "vi2" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR-NEXT: %[[U1:.*]] = cir.alloca "u1" align(4) init : !cir.ptr<!u32i>
  // CIR-NEXT: %[[U2:.*]] = cir.alloca "u2" align(4) init : !cir.ptr<!u32i>
  // CIR-NEXT: %[[VU1:.*]] = cir.alloca "vu1" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR-NEXT: %[[VU2:.*]] = cir.alloca "vu2" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR-NEXT: %[[BI1:.*]] = cir.alloca "bi1" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR-NEXT: %[[BI2:.*]] = cir.alloca "bi2" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR-NEXT: %[[BU1:.*]] = cir.alloca "bu1" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR-NEXT: %[[BU1:.*]] = cir.alloca "bu2" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR-NEXT: %[[CVF1:.*]] = cir.alloca "cvf1" align(16) init const : !cir.ptr<!cir.vector<4 x !cir.float>>

  // LLVM: %[[ADDR_F1:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_F2:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_D1:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_D2:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_VF1:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_VF2:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_I1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_I2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_VI1:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_VI2:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_U1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_U2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_VU1:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_VU2:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_BI1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BI2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BU1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_BU2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_CVF1:.*]] = alloca <4 x float>, align 16

  // CIR:      %[[F1_LOAD:.*]] = cir.load align(4) %[[F1]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: %[[F2_LOAD:.*]] = cir.load align(4) %[[F2]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: cir.call_llvm_intrinsic "minimum" %[[F1_LOAD]], %[[F2_LOAD]] : (!cir.float, !cir.float) -> !cir.float

  // LLVM:      [[F1:%.+]] = load float, ptr %[[ADDR_F1]], align 4
  // LLVM-NEXT: [[F2:%.+]] = load float, ptr %[[ADDR_F2]], align 4
  // LLVM-NEXT:  call float @llvm.minimum.f32(float [[F1]], float [[F2]])
  f1 = __builtin_elementwise_minimum(f1, f2);

  // CIR:      %[[D1_LOAD:.*]] = cir.load align(8) %[[D1]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: %[[D2_LOAD:.*]] = cir.load align(8) %[[D2]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "minimum" %[[D1_LOAD]], %[[D2_LOAD]] : (!cir.double, !cir.double) -> !cir.double

  // LLVM:      [[D1:%.+]] = load double, ptr %[[ADDR_D1]], align 8
  // LLVM-NEXT: [[D2:%.+]] = load double, ptr %[[ADDR_D2]], align 8
  // LLVM-NEXT: call double @llvm.minimum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_minimum(d1, d2);

  // CIR:      %[[D1_LOAD:.*]] = cir.load align(8) %[[D1]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: %[[TWO:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "minimum" %[[D1_LOAD]], %[[TWO]] : (!cir.double, !cir.double) -> !cir.double

  // LLVM:      [[D1:%.+]] = load double, ptr %[[ADDR_D1]], align 8
  // LLVM-NEXT: call double @llvm.minimum.f64(double [[D1]], double 2.000000e+00)
  d1 = __builtin_elementwise_minimum(d1, 2.0);

  // CIR:      %[[VF1_LOAD:.*]] = cir.load align(16) %[[VF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "minimum" %[[VF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF1:%.+]] = load <4 x float>, ptr %[[ADDR_VF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.minimum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_minimum(vf1, vf2);

  // CIR:      %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "minimum" %[[CVF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.minimum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_minimum(cvf1, vf2);

  // CIR:      %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "minimum" %[[VF2_LOAD]], %[[CVF1_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.minimum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_minimum(vf2, cvf1);
}

void test_builtin_elementwise_max(float f1, float f2, double d1, double d2,
                                  float4 vf1, float4 vf2, long long int i1,
                                  long long int i2, si8 vi1, si8 vi2,
                                  unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                  _BitInt(31) bi1, _BitInt(31) bi2,
                                  unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CIR-LABEL: @test_builtin_elementwise_max
  // LLVM-LABEL: @test_builtin_elementwise_max(

  // CIR: %[[F1:.*]] = cir.alloca "f1" align(4) init : !cir.ptr<!cir.float>
  // CIR: %[[F2:.*]] = cir.alloca "f2" align(4) init : !cir.ptr<!cir.float>
  // CIR: %[[D1:.*]] = cir.alloca "d1" align(8) init : !cir.ptr<!cir.double>
  // CIR: %[[D2:.*]] = cir.alloca "d2" align(8) init : !cir.ptr<!cir.double>
  // CIR: %[[VF1:.*]] = cir.alloca "vf1" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR: %[[VF2:.*]] = cir.alloca "vf2" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR: %[[I1:.*]] = cir.alloca "i1" align(8) init : !cir.ptr<!s64i>
  // CIR: %[[I2:.*]] = cir.alloca "i2" align(8) init : !cir.ptr<!s64i>
  // CIR: %[[VI1:.*]] = cir.alloca "vi1" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR: %[[VI2:.*]] = cir.alloca "vi2" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR: %[[U1:.*]] = cir.alloca "u1" align(4) init : !cir.ptr<!u32i>
  // CIR: %[[U2:.*]] = cir.alloca "u2" align(4) init : !cir.ptr<!u32i>
  // CIR: %[[VU1:.*]] = cir.alloca "vu1" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR: %[[VU2:.*]] = cir.alloca "vu2" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR: %[[BI1:.*]] = cir.alloca "bi1" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR: %[[BI2:.*]] = cir.alloca "bi2" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR: %[[BU1:.*]] = cir.alloca "bu1" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR: %[[BU2:.*]] = cir.alloca "bu2" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR: %[[CVF1:.*]] = cir.alloca "cvf1" align(16) init const : !cir.ptr<!cir.vector<4 x !cir.float>>

  // LLVM: %[[ADDR_F1:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_F2:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_D1:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_D2:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_VF1:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_VF2:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_I1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_I2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_VI1:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_VI2:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_U1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_U2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_VU1:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_VU2:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_BI1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BI2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BU1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_BU2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_CVF1:.*]] = alloca <4 x float>, align 16

  // CIR:      %[[F1_LOAD:.*]] = cir.load align(4) %[[F1]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: %[[F2_LOAD:.*]] = cir.load align(4) %[[F2]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: cir.call_llvm_intrinsic "maxnum" %[[F1_LOAD]], %[[F2_LOAD]] : (!cir.float, !cir.float) -> !cir.float
  
  // LLVM:      [[F1:%.+]] = load float, ptr %[[ADDR_F1]], align 4
  // LLVM-NEXT: [[F2:%.+]] = load float, ptr %[[ADDR_F2]], align 4
  // LLVM-NEXT:  call float @llvm.maxnum.f32(float [[F1]], float [[F2]])
  f1 = __builtin_elementwise_max(f1, f2);

  // CIR:      %[[D1_LOAD:.*]] = cir.load align(8) %[[D1]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: %[[D2_LOAD:.*]] = cir.load align(8) %[[D2]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "maxnum" %[[D1_LOAD]], %[[D2_LOAD]] : (!cir.double, !cir.double) -> !cir.double

  // LLVM:      [[D1:%.+]] = load double, ptr %[[ADDR_D1]], align 8
  // LLVM-NEXT: [[D2:%.+]] = load double, ptr %[[ADDR_D2]], align 8
  // LLVM-NEXT: call double @llvm.maxnum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_max(d1, d2);

  // CIR:      %[[TWENTY:.*]] = cir.const #cir.fp<2.000000e+01> : !cir.double
  // CIR-NEXT: %[[D2_LOAD:.*]] = cir.load align(8) %[[D2]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "maxnum" %[[TWENTY]], %[[D2_LOAD]] : (!cir.double, !cir.double) -> !cir.double

  // LLVM:      [[D2:%.+]] = load double, ptr %[[ADDR_D2]], align 8
  // LLVM-NEXT: call double @llvm.maxnum.f64(double 2.000000e+01, double [[D2]])
  d1 = __builtin_elementwise_max(20.0, d2);

  // CIR:      %[[VF1_LOAD:.*]] = cir.load align(16) %[[VF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "maxnum" %[[VF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF1:%.+]] = load <4 x float>, ptr %[[ADDR_VF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_max(vf1, vf2);

  // CIR:      %[[I1_LOAD:.*]] = cir.load align(8) %[[I1]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: %[[I2_LOAD:.*]] = cir.load align(8) %[[I2]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: cir.call_llvm_intrinsic "smax" %31, %32 : (!s64i, !s64i) -> !s64i

  // LLVM:      [[I1:%.+]] = load i64, ptr %[[ADDR_I1]], align 8
  // LLVM-NEXT: [[I2:%.+]] = load i64, ptr %[[ADDR_I2]], align 8
  // LLVM-NEXT: call i64 @llvm.smax.i64(i64 [[I1]], i64 [[I2]])
  i1 = __builtin_elementwise_max(i1, i2);

  // CIR:      %[[I1_LOAD:.*]] = cir.load align(8) %[[I1]] : !cir.ptr<!s64i>, !s64i
  // CIR-NEXT: %[[TEN:.*]] = cir.const #cir.int<10> : !s64i
  // CIR-NEXT: cir.call_llvm_intrinsic "smax" %[[I1_LOAD]], %[[TEN]] : (!s64i, !s64i) -> !s64i

  // LLVM:      [[I1:%.+]] = load i64, ptr %[[ADDR_I1]], align 8
  // LLVM-NEXT: call i64 @llvm.smax.i64(i64 [[I1]], i64 10)
  i1 = __builtin_elementwise_max(i1, 10ll);

  // CIR:      %[[VI1_LOAD:.*]] = cir.load align(16) %[[VI1]] : !cir.ptr<!cir.vector<8 x !s16i>>, !cir.vector<8 x !s16i>
  // CIR-NEXT: %[[VI2_LOAD:.*]] = cir.load align(16) %[[VI2]] : !cir.ptr<!cir.vector<8 x !s16i>>, !cir.vector<8 x !s16i>
  // CIR-NEXT: cir.call_llvm_intrinsic "smax" %[[VI1_LOAD]], %[[VI2_LOAD]] : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>

  // LLVM:      [[VI1:%.+]] = load <8 x i16>, ptr %[[ADDR_VI1]], align 16
  // LLVM-NEXT: [[VI2:%.+]] = load <8 x i16>, ptr %[[ADDR_VI2]], align 16
  // LLVM-NEXT: call <8 x i16> @llvm.smax.v8i16(<8 x i16> [[VI1]], <8 x i16> [[VI2]])
  vi1 = __builtin_elementwise_max(vi1, vi2);

  // CIR:      %[[U1_LOAD:.*]] = cir.load align(4) %[[U1]] : !cir.ptr<!u32i>, !u32i
  // CIR-NEXT: %[[U2_LOAD:.*]] = cir.load align(4) %[[U2]] : !cir.ptr<!u32i>, !u32i
  // CIR-NEXT: cir.call_llvm_intrinsic "umax" %[[U1_LOAD]], %[[U2_LOAD]] : (!u32i, !u32i) -> !u32i

  // LLVM:      [[U1:%.+]] = load i32, ptr %[[ADDR_U1]], align 4
  // LLVM-NEXT: [[U2:%.+]] = load i32, ptr %[[ADDR_U2]], align 4
  // LLVM-NEXT: call i32 @llvm.umax.i32(i32 [[U1]], i32 [[U2]])
  u1 = __builtin_elementwise_max(u1, u2);

  // CIR:      %[[VU1_LOAD:.*]] = cir.load align(16) %[[VU1]] : !cir.ptr<!cir.vector<4 x !u32i>>, !cir.vector<4 x !u32i>
  // CIR-NEXT: %[[VU2_LOAD:.*]] = cir.load align(16) %[[VU2]] : !cir.ptr<!cir.vector<4 x !u32i>>, !cir.vector<4 x !u32i>
  // CIR-NEXT: cir.call_llvm_intrinsic "umax" %[[VU1_LOAD]], %[[VU2_LOAD]] : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>) -> !cir.vector<4 x !u32i>

  // LLVM:      [[VU1:%.+]] = load <4 x i32>, ptr %[[ADDR_VU1]], align 16
  // LLVM-NEXT: [[VU2:%.+]] = load <4 x i32>, ptr %[[ADDR_VU2]], align 16
  // LLVM-NEXT: call <4 x i32> @llvm.umax.v4i32(<4 x i32> [[VU1]], <4 x i32> [[VU2]])
  vu1 = __builtin_elementwise_max(vu1, vu2);

  // CIR:      %[[BI1_LOAD:.*]] = cir.load align(4) %[[BI1]] : !cir.ptr<!cir.int<s, 31, bitint>>, !cir.int<s, 31, bitint>
  // CIR-NEXT: %[[BI2_LOAD:.*]] = cir.load align(4) %[[BI2]] : !cir.ptr<!cir.int<s, 31, bitint>>, !cir.int<s, 31, bitint>
  // CIR-NEXT: cir.call_llvm_intrinsic "smax" %[[BI1_LOAD]], %[[BI2_LOAD]] : (!cir.int<s, 31, bitint>, !cir.int<s, 31, bitint>) -> !cir.int<s, 31, bitint>

  // LLVM:      [[BI1:%.+]] = load i32, ptr %[[ADDR_BI1]], align 4
  // LLVM-NEXT: [[LOADEDV:%.+]] = trunc i32 [[BI1]] to i31
  // LLVM-NEXT: [[BI2:%.+]] = load i32, ptr %[[ADDR_BI2]], align 4
  // LLVM-NEXT: [[LOADEDV1:%.+]] = trunc i32 [[BI2]] to i31
  // LLVM-NEXT: call i31 @llvm.smax.i31(i31 [[LOADEDV]], i31 [[LOADEDV1]])
  bi1 = __builtin_elementwise_max(bi1, bi2);

  // CIR:      %[[BU1_LOAD:.*]] = cir.load align(8) %[[BU1]] : !cir.ptr<!cir.int<u, 55, bitint>>, !cir.int<u, 55, bitint>
  // CIR-NEXT: %[[BU2_LOAD:.*]] = cir.load align(8) %[[BU2]] : !cir.ptr<!cir.int<u, 55, bitint>>, !cir.int<u, 55, bitint>
  // CIR-NEXT: cir.call_llvm_intrinsic "umax" %[[BU1_LOAD]], %[[BU2_LOAD]] : (!cir.int<u, 55, bitint>, !cir.int<u, 55, bitint>) -> !cir.int<u, 55, bitint>

  // LLVM:      [[BU1:%.+]] = load i64, ptr %[[ADDR_BU1]], align 8
  // LLVM-NEXT: [[LOADEDV2:%.+]] = trunc i64 [[BU1]] to i55
  // LLVM-NEXT: [[BU2:%.+]] = load i64, ptr %[[ADDR_BU2]], align 8
  // LLVM-NEXT: [[LOADEDV3:%.+]] = trunc i64 [[BU2]] to i55
  // LLVM-NEXT: call i55 @llvm.umax.i55(i55 [[LOADEDV2]], i55 [[LOADEDV3]])
  bu1 = __builtin_elementwise_max(bu1, bu2);
 
  // CIR:      %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "maxnum" %[[CVF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_max(cvf1, vf2);

  // CIR:      %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "maxnum" %[[VF2_LOAD]], %[[CVF1_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.maxnum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_max(vf2, cvf1);

  // CIR:      %[[IAONE:.*]] = cir.get_global @int_as_one : !cir.ptr<!s32i, target_address_space(1)>
  // CIR-NEXT: %[[IAO_LOAD:.*]] = cir.load align(4) %[[IAONE]] : !cir.ptr<!s32i, target_address_space(1)>, !s32i
  // CIR-NEXT: %[[B:.*]] = cir.get_global @b : !cir.ptr<!s32i>
  // CIR-NEXT: %[[B_LOAD:.*]] = cir.load align(4) %[[B]] : !cir.ptr<!s32i>, !s32i
  // CIR-NEXT: cir.call_llvm_intrinsic "smax" %[[IAO_LOAD]], %[[B_LOAD]] : (!s32i, !s32i) -> !s32i

  // LLVM:      [[IAS1:%.+]] = load i32, ptr addrspace(1) @int_as_one, align 4
  // LLVM-NEXT: [[B:%.+]] = load i32, ptr @b, align 4
  // LLVM-NEXT: call i32 @llvm.smax.i32(i32 [[IAS1]], i32 [[B]])
  int_as_one = __builtin_elementwise_max(int_as_one, b);

  // CIR: %[[A_CONST:.*]] = cir.const #cir.int<97> : !s64i
  // CIR-NEXT: cir.store align(8) %[[A_CONST]], %[[I1]] : !s64i, !cir.ptr<!s64i>

  // LLVM: store i64 97, ptr [[I1:%.+]], align 8
  i1 = __builtin_elementwise_max(1, 'a');
}

void test_builtin_elementwise_maximum(float f1, float f2, double d1, double d2,
                                      float4 vf1, float4 vf2, long long int i1,
                                      long long int i2, si8 vi1, si8 vi2,
                                      unsigned u1, unsigned u2, u4 vu1, u4 vu2,
                                      _BitInt(31) bi1, _BitInt(31) bi2,
                                      unsigned _BitInt(55) bu1, unsigned _BitInt(55) bu2) {
  // CIR-LABEL: test_builtin_elementwise_maximum(
  // LLVM-LABEL: @test_builtin_elementwise_maximum(

  // CIR-NEXT: %[[F1:.*]] = cir.alloca "f1" align(4) init : !cir.ptr<!cir.float>
  // CIR-NEXT: %[[F2:.*]] = cir.alloca "f2" align(4) init : !cir.ptr<!cir.float>
  // CIR-NEXT: %[[D1:.*]] = cir.alloca "d1" align(8) init : !cir.ptr<!cir.double>
  // CIR-NEXT: %[[D2:.*]] = cir.alloca "d2" align(8) init : !cir.ptr<!cir.double>
  // CIR-NEXT: %[[VF1:.*]] = cir.alloca "vf1" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR-NEXT: %[[VF2:.*]] = cir.alloca "vf2" align(16) init : !cir.ptr<!cir.vector<4 x !cir.float>>
  // CIR-NEXT: %[[I1:.*]] = cir.alloca "i1" align(8) init : !cir.ptr<!s64i>
  // CIR-NEXT: %[[I1:.*]] = cir.alloca "i2" align(8) init : !cir.ptr<!s64i>
  // CIR-NEXT: %[[VI1:.*]] = cir.alloca "vi1" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR-NEXT: %[[VI2:.*]] = cir.alloca "vi2" align(16) init : !cir.ptr<!cir.vector<8 x !s16i>>
  // CIR-NEXT: %[[U1:.*]] = cir.alloca "u1" align(4) init : !cir.ptr<!u32i>
  // CIR-NEXT: %[[U2:.*]] = cir.alloca "u2" align(4) init : !cir.ptr<!u32i>
  // CIR-NEXT: %[[VU1:.*]] = cir.alloca "vu1" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR-NEXT: %[[VU2:.*]] = cir.alloca "vu2" align(16) init : !cir.ptr<!cir.vector<4 x !u32i>>
  // CIR-NEXT: %[[BI1:.*]] = cir.alloca "bi1" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR-NEXT: %[[BI2:.*]] = cir.alloca "bi2" align(4) init : !cir.ptr<!cir.int<s, 31, bitint>>
  // CIR-NEXT: %[[BU1:.*]] = cir.alloca "bu1" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR-NEXT: %[[BU2:.*]] = cir.alloca "bu2" align(8) init : !cir.ptr<!cir.int<u, 55, bitint>>
  // CIR-NEXT: %[[CVF1:.*]] = cir.alloca "cvf1" align(16) init const : !cir.ptr<!cir.vector<4 x !cir.float>>

  // LLVM: %[[ADDR_F1:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_F2:.*]] = alloca float, align 4
  // LLVM: %[[ADDR_D1:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_D2:.*]] = alloca double, align 8
  // LLVM: %[[ADDR_VF1:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_VF2:.*]] = alloca <4 x float>, align 16
  // LLVM: %[[ADDR_I1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_I2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_VI1:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_VI2:.*]] = alloca <8 x i16>, align 16
  // LLVM: %[[ADDR_U1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_U2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_VU1:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_VU2:.*]] = alloca <4 x i32>, align 16
  // LLVM: %[[ADDR_BI1:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BI2:.*]] = alloca i32, align 4
  // LLVM: %[[ADDR_BU1:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_BU2:.*]] = alloca i64, align 8
  // LLVM: %[[ADDR_CVF1:.*]] = alloca <4 x float>, align 16

  // CIR:      %[[F1_LOAD:.*]] = cir.load align(4) %[[F1]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: %[[F2_LOAD:.*]] = cir.load align(4) %[[F2]] : !cir.ptr<!cir.float>, !cir.float
  // CIR-NEXT: cir.call_llvm_intrinsic "maximum" %[[F1_LOAD]], %[[F2_LOAD]] : (!cir.float, !cir.float) -> !cir.float

  // LLVM:      [[F1:%.+]] = load float, ptr %[[ADDR_F1]], align 4
  // LLVM-NEXT: [[F2:%.+]] = load float, ptr %[[ADDR_F2]], align 4
  // LLVM-NEXT:  call float @llvm.maximum.f32(float [[F1]], float [[F2]])
  f1 = __builtin_elementwise_maximum(f1, f2);

  // CIR:      %[[D1_LOAD:.*]] = cir.load align(8) %[[D1]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: %[[D2_LOAD:.*]] = cir.load align(8) %[[D2]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "maximum" %[[D1_LOAD]], %[[D2_LOAD]] : (!cir.double, !cir.double) -> !cir.double

  // LLVM:      [[D1:%.+]] = load double, ptr %[[ADDR_D1]], align 8
  // LLVM-NEXT: [[D2:%.+]] = load double, ptr %[[ADDR_D2]], align 8
  // LLVM-NEXT: call double @llvm.maximum.f64(double [[D1]], double [[D2]])
  d1 = __builtin_elementwise_maximum(d1, d2);

  // CIR:      %[[TWENTY:.*]] = cir.const #cir.fp<2.000000e+01> : !cir.double
  // CIR-NEXT: %[[D2_LOAD:.*]] = cir.load align(8) %[[D2]] : !cir.ptr<!cir.double>, !cir.double
  // CIR-NEXT: cir.call_llvm_intrinsic "maximum" %[[TWENTY]], %[[D2_LOAD]] : (!cir.double, !cir.double) -> !cir.double

  // LLVM:      [[D2:%.+]] = load double, ptr %[[ADDR_D2]], align 8
  // LLVM-NEXT: call double @llvm.maximum.f64(double 2.000000e+01, double [[D2]])
  d1 = __builtin_elementwise_maximum(20.0, d2);

  // CIR:      %[[VF1_LOAD:.*]] = cir.load align(16) %[[VF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "maximum" %[[VF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF1:%.+]] = load <4 x float>, ptr %[[ADDR_VF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.maximum.v4f32(<4 x float> [[VF1]], <4 x float> [[VF2]])
  vf1 = __builtin_elementwise_maximum(vf1, vf2);

  // CIR:      %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "maximum" %[[CVF1_LOAD]], %[[VF2_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.maximum.v4f32(<4 x float> [[CVF1]], <4 x float> [[VF2]])
  const float4 cvf1 = vf1;
  vf1 = __builtin_elementwise_maximum(cvf1, vf2);

  // CIR:      %[[VF2_LOAD:.*]] = cir.load align(16) %[[VF2]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: %[[CVF1_LOAD:.*]] = cir.load align(16) %[[CVF1]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR-NEXT: cir.call_llvm_intrinsic "maximum" %[[VF2_LOAD]], %[[CVF1_LOAD]] : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM:      [[VF2:%.+]] = load <4 x float>, ptr %[[ADDR_VF2]], align 16
  // LLVM-NEXT: [[CVF1:%.+]] = load <4 x float>, ptr %[[ADDR_CVF1]], align 16
  // LLVM-NEXT: call <4 x float> @llvm.maximum.v4f32(<4 x float> [[VF2]], <4 x float> [[CVF1]])
  vf1 = __builtin_elementwise_maximum(vf2, cvf1);
}
