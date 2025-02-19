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
  // CIR: {{%.*}} = cir.llvm.intrinsic "acos" {{%.*}} : (!cir.float) -> !cir.float
  // LLVM: {{%.*}} = call float @llvm.acos.f32(float {{%.*}})
  f = __builtin_elementwise_acos(f);

  // CIR: {{%.*}} = cir.llvm.intrinsic "acos" {{%.*}} : (!cir.double) -> !cir.double
  // LLVM: {{%.*}} = call double @llvm.acos.f64(double {{%.*}})
  d = __builtin_elementwise_acos(d);

  // CIR: {{%.*}} = cir.llvm.intrinsic "acos" {{%.*}} : (!cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>
  // LLVM: {{%.*}} = call <4 x float> @llvm.acos.v4f32(<4 x float> {{%.*}})
  vf4 = __builtin_elementwise_acos(vf4);

  // CIR: {{%.*}} = cir.llvm.intrinsic "acos" {{%.*}} : (!cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>
  // LLVM: {{%.*}} = call <4 x double> @llvm.acos.v4f64(<4 x double> {{%.*}})
  vd4 = __builtin_elementwise_acos(vd4);
}
