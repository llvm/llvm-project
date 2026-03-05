// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -emit-cir %s -o %t.cir  
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:  -emit-llvm  %s -o %t.ll
// RUN: FileCheck  --check-prefix=LLVM --input-file=%t.ll %s

typedef int vint4 __attribute__((ext_vector_type(4)));
typedef float vfloat4 __attribute__((ext_vector_type(4)));
typedef double vdouble4 __attribute__((ext_vector_type(4)));

void test_builtin_elementwise_abs(vint4 vi4, int i, float f, double d, 
                                  vfloat4 vf4, vdouble4  vd4) {
    // CIR-LABEL: test_builtin_elementwise_abs
    // LLVM-LABEL: test_builtin_elementwise_abs
    // CIR: {{%.*}} = cir.fabs {{%.*}} : !cir.float
    // LLVM: {{%.*}} = call float @llvm.fabs.f32(float {{%.*}})
    f = __builtin_elementwise_abs(f);

    // CIR: {{%.*}} = cir.fabs {{%.*}} : !cir.double
    // LLVM: {{%.*}} = call double @llvm.fabs.f64(double {{%.*}})
    d = __builtin_elementwise_abs(d);

    // CIR: {{%.*}} = cir.abs {{%.*}} : !cir.vector<!s32i x 4>
    // LLVM: {{%.*}} = call <4 x i32> @llvm.abs.v4i32(<4 x i32> {{%.*}}, i1 false)
    vi4 = __builtin_elementwise_abs(vi4);

    // CIR: {{%.*}} = cir.abs {{%.*}} : !s32
    // LLVM: {{%.*}} = call i32 @llvm.abs.i32(i32 {{%.*}}, i1 false)
    i = __builtin_elementwise_abs(i);

    // CIR: {{%.*}} = cir.fabs {{%.*}} : !cir.vector<!cir.float x 4>
    // LLVM: {{%.*}} = call <4 x float> @llvm.fabs.v4f32(<4 x float> {{%.*}})
    vf4 = __builtin_elementwise_abs(vf4);

    // CIR: {{%.*}} = cir.fabs {{%.*}} : !cir.vector<!cir.double x 4>
    // LLVM: {{%.*}} = call <4 x double> @llvm.fabs.v4f64(<4 x double> {{%.*}})
    vd4 = __builtin_elementwise_abs(vd4);
}

void test_builtin_elementwise_acos(float f, double d, vfloat4 vf4,
                                   vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_acos
  // LLVM-LABEL: test_builtin_elementwise_acos
  // CIR: {{%.*}} = cir.acos {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.acos.f32(float {{%.*}})
  f = __builtin_elementwise_acos(f);

  // CIR: {{%.*}} = cir.acos {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.acos.f64(double {{%.*}})
  d = __builtin_elementwise_acos(d);

  // CIR: {{%.*}} = cir.acos {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.acos.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_acos(vf4);

  // CIR: {{%.*}} = cir.acos {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.acos.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_acos(vd4);
}

void test_builtin_elementwise_asin(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_asin
  // LLVM-LABEL: test_builtin_elementwise_asin
  // CIR: {{%.*}} = cir.asin {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.asin.f32(float {{%.*}})
  f = __builtin_elementwise_asin(f);

  // CIR: {{%.*}} = cir.asin {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.asin.f64(double {{%.*}})
  d = __builtin_elementwise_asin(d);

  // CIR: {{%.*}} = cir.asin {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.asin.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_asin(vf4);

  // CIR: {{%.*}} = cir.asin {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.asin.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_asin(vd4);
}

void test_builtin_elementwise_atan(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_atan
  // LLVM-LABEL: test_builtin_elementwise_atan
  // CIR: {{%.*}} = cir.atan {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.atan.f32(float {{%.*}})
  f = __builtin_elementwise_atan(f);

  // CIR: {{%.*}} = cir.atan {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.atan.f64(double {{%.*}})
  d = __builtin_elementwise_atan(d);

  // CIR: {{%.*}} = cir.atan {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.atan.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_atan(vf4);

  // CIR: {{%.*}} = cir.atan {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.atan.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_atan(vd4);
}

void test_builtin_elementwise_atan2(float f, double d, vfloat4 vf4,
  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_atan2
  // LLVM-LABEL: test_builtin_elementwise_atan2
  // CIR: {{%.*}} = cir.atan2 {{%.*}}, {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.atan2.f32(float {{%.*}}, float {{%.*}})
  f = __builtin_elementwise_atan2(f, f);

  // CIR: {{%.*}} = cir.atan2 {{%.*}}, {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.atan2.f64(double {{%.*}}, double {{%.*}})
  d = __builtin_elementwise_atan2(d, d);

  // CIR: {{%.*}} = cir.atan2 {{%.*}}, {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.atan2.v4f32(<4 x float> {{%.*}}, <4 x float> {{%.*}})
  vf4 = __builtin_elementwise_atan2(vf4, vf4);

  // CIR: {{%.*}} = cir.atan2 {{%.*}}, {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.atan2.v4f64(<4 x double> {{%.*}}, <4 x double> {{%.*}})
  vd4 = __builtin_elementwise_atan2(vd4, vd4);
}

void test_builtin_elementwise_exp(float f, double d, vfloat4 vf4,
                                  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_exp
  // LLVM-LABEL: test_builtin_elementwise_exp
  // CIR: {{%.*}} = cir.exp {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.exp.f32(float {{%.*}})
  f = __builtin_elementwise_exp(f);

  // CIR: {{%.*}} = cir.exp {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.exp.f64(double {{%.*}})
  d = __builtin_elementwise_exp(d);

  // CIR: {{%.*}} = cir.exp {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.exp.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_exp(vf4);

  // CIR: {{%.*}} = cir.exp {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.exp.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_exp(vd4);
}

void test_builtin_elementwise_exp2(float f, double d, vfloat4 vf4,
                                  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_exp
  // LLVM-LABEL: test_builtin_elementwise_exp
  // CIR: {{%.*}} = cir.exp2 {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.exp2.f32(float {{%.*}})
  f = __builtin_elementwise_exp2(f);

  // CIR: {{%.*}} = cir.exp2 {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.exp2.f64(double {{%.*}})
  d = __builtin_elementwise_exp2(d);

  // CIR: {{%.*}} = cir.exp2 {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.exp2.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_exp2(vf4);

  // CIR: {{%.*}} = cir.exp2 {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.exp2.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_exp2(vd4);
}

void test_builtin_elementwise_log(float f, double d, vfloat4 vf4,
                                  vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_log
  // LLVM-LABEL: test_builtin_elementwise_log
  // CIR: {{%.*}} = cir.log {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.log.f32(float {{%.*}})
  f = __builtin_elementwise_log(f);

  // CIR: {{%.*}} = cir.log {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.log.f64(double {{%.*}})
  d = __builtin_elementwise_log(d);

  // CIR: {{%.*}} = cir.log {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.log.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_log(vf4);

  // CIR: {{%.*}} = cir.log {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.log.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_log(vd4);
}

void test_builtin_elementwise_log2(float f, double d, vfloat4 vf4,
                                    vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_log2
  // LLVM-LABEL: test_builtin_elementwise_log2
  // CIR: {{%.*}} = cir.log2 {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.log2.f32(float {{%.*}})
  f = __builtin_elementwise_log2(f);

  // CIR: {{%.*}} = cir.log2 {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.log2.f64(double {{%.*}})
  d = __builtin_elementwise_log2(d);

  // CIR: {{%.*}} = cir.log2 {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.log2.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_log2(vf4);

  // CIR: {{%.*}} = cir.log2 {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.log2.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_log2(vd4);
}

void test_builtin_elementwise_log10(float f, double d, vfloat4 vf4,
                                     vdouble4  vd4) {
  // CIR-LABEL: test_builtin_elementwise_log10
  // LLVM-LABEL: test_builtin_elementwise_log10
  // CIR: {{%.*}} = cir.log10 {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.log10.f32(float {{%.*}})
  f = __builtin_elementwise_log10(f);

  // CIR: {{%.*}} = cir.log10 {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.log10.f64(double {{%.*}})
  d = __builtin_elementwise_log10(d);

  // CIR: {{%.*}} = cir.log10 {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.log10.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_log10(vf4);

  // CIR: {{%.*}} = cir.log10 {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.log10.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_log10(vd4);
}

void test_builtin_elementwise_cos(float f, double d, vfloat4 vf4,
                                     vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_cos
  // LLVM-LABEL: test_builtin_elementwise_cos
  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.cos.f32(float {{%.*}})
  f = __builtin_elementwise_cos(f);

  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.cos.f64(double {{%.*}})
  d = __builtin_elementwise_cos(d);

  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.cos.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_cos(vf4);

  // CIR: {{%.*}} = cir.cos {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.cos.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_cos(vd4);
}

void test_builtin_elementwise_floor(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_floor
  // LLVM-LABEL: test_builtin_elementwise_floor
  // CIR: {{%.*}} = cir.floor {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.floor.f32(float {{%.*}})
  f = __builtin_elementwise_floor(f);

  // CIR: {{%.*}} = cir.floor {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.floor.f64(double {{%.*}})
  d = __builtin_elementwise_floor(d);

  // CIR: {{%.*}} = cir.floor {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.floor.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_floor(vf4);

  // CIR: {{%.*}} = cir.floor {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.floor.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_floor(vd4);
}

void test_builtin_elementwise_round(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_round
  // LLVM-LABEL: test_builtin_elementwise_round
  // CIR: {{%.*}} = cir.round {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.round.f32(float {{%.*}})
  f = __builtin_elementwise_round(f);

  // CIR: {{%.*}} = cir.round {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.round.f64(double {{%.*}})
  d = __builtin_elementwise_round(d);

  // CIR: {{%.*}} = cir.round {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.round.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_round(vf4);

  // CIR: {{%.*}} = cir.round {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.round.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_round(vd4);
}

void test_builtin_elementwise_rint(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_rint
  // LLVM-LABEL: test_builtin_elementwise_rint
  // CIR: {{%.*}} = cir.rint {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.rint.f32(float {{%.*}})
  f = __builtin_elementwise_rint(f);

  // CIR: {{%.*}} = cir.rint {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.rint.f64(double {{%.*}})
  d = __builtin_elementwise_rint(d);

  // CIR: {{%.*}} = cir.rint {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.rint.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_rint(vf4);

  // CIR: {{%.*}} = cir.rint {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.rint.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_rint(vd4);
}

void test_builtin_elementwise_nearbyint(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_nearbyint
  // LLVM-LABEL: test_builtin_elementwise_nearbyint
  // CIR: {{%.*}} = cir.nearbyint {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.nearbyint.f32(float {{%.*}})
  f = __builtin_elementwise_nearbyint(f);

  // CIR: {{%.*}} = cir.nearbyint {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.nearbyint.f64(double {{%.*}})
  d = __builtin_elementwise_nearbyint(d);

  // CIR: {{%.*}} = cir.nearbyint {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_nearbyint(vf4);

  // CIR: {{%.*}} = cir.nearbyint {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_nearbyint(vd4);
}

void test_builtin_elementwise_sin(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_sin
  // LLVM-LABEL: test_builtin_elementwise_sin
  // CIR: {{%.*}} = cir.sin {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.sin.f32(float {{%.*}})
  f = __builtin_elementwise_sin(f);

  // CIR: {{%.*}} = cir.sin {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.sin.f64(double {{%.*}})
  d = __builtin_elementwise_sin(d);

  // CIR: {{%.*}} = cir.sin {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.sin.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_sin(vf4);

  // CIR: {{%.*}} = cir.sin {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.sin.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_sin(vd4);
}

void test_builtin_elementwise_sqrt(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_sqrt
  // LLVM-LABEL: test_builtin_elementwise_sqrt
  // CIR: {{%.*}} = cir.sqrt {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.sqrt.f32(float {{%.*}})
  f = __builtin_elementwise_sqrt(f);

  // CIR: {{%.*}} = cir.sqrt {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.sqrt.f64(double {{%.*}})
  d = __builtin_elementwise_sqrt(d);

  // CIR: {{%.*}} = cir.sqrt {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.sqrt.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_sqrt(vf4);

  // CIR: {{%.*}} = cir.sqrt {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.sqrt.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_sqrt(vd4);
}

void test_builtin_elementwise_tan(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_tan
  // LLVM-LABEL: test_builtin_elementwise_tan
  // CIR: {{%.*}} = cir.tan {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.tan.f32(float {{%.*}})
  f = __builtin_elementwise_tan(f);

  // CIR: {{%.*}} = cir.tan {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.tan.f64(double {{%.*}})
  d = __builtin_elementwise_tan(d);

  // CIR: {{%.*}} = cir.tan {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.tan.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_tan(vf4);

  // CIR: {{%.*}} = cir.tan {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.tan.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_tan(vd4);
}

void test_builtin_elementwise_trunc(float f, double d, vfloat4 vf4,
                   vdouble4 vd4) {
  // CIR-LABEL: test_builtin_elementwise_trunc
  // LLVM-LABEL: test_builtin_elementwise_trunc
  // CIR: {{%.*}} = cir.trunc {{%.*}} : !cir.float
  // LLVM: {{%.*}} = call float @llvm.trunc.f32(float {{%.*}})
  f = __builtin_elementwise_trunc(f);

  // CIR: {{%.*}} = cir.trunc {{%.*}} : !cir.double
  // LLVM: {{%.*}} = call double @llvm.trunc.f64(double {{%.*}})
  d = __builtin_elementwise_trunc(d);

  // CIR: {{%.*}} = cir.trunc {{%.*}} : !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.trunc.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_trunc(vf4);

  // CIR: {{%.*}} = cir.trunc {{%.*}} : !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.trunc.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_trunc(vd4);
}
