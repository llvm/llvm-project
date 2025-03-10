// RUN: %clang_cc1 -no-enable-noundef-analysis -triple spirv-unknown-vulkan-compute -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple dxil-pc-shadermodel6.3-compute -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Tests the matrix type transformation builtin.

// CHECK-LABEL: define {{.*}}transpose_double_4x4
void transpose_double_4x4(double4x4 a) {
  // CHECK:       [[A:%.*]] = load <16 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:   [[TRANS:%.*]] = call <16 x double> @llvm.matrix.transpose.v16f64(<16 x double> [[A]], i32 4, i32 4)
  // CHECK-NEXT:  store <16 x double> [[TRANS]], ptr %a_t, align 8

  double4x4 a_t = __builtin_matrix_transpose(a);
}

// CHECK-LABEL: define {{.*}}transpose_float_3x2
void transpose_float_3x2(float3x2 a) {
  // CHECK:        [[A:%.*]] = load <6 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:   [[TRANS:%.*]] = call <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[A]], i32 3, i32 2)
  // CHECK-NEXT:   store <6 x float> [[TRANS]], ptr %a_t, align 4

  float2x3 a_t = __builtin_matrix_transpose(a);
}

// CHECK-LABEL: define {{.*}}transpose_int_4x3
void transpose_int_4x3(int4x3 a) {
  // CHECK:         [[A:%.*]] = load <12 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:    [[TRANS:%.*]] = call <12 x i32> @llvm.matrix.transpose.v12i32(<12 x i32> [[A]], i32 4, i32 3)
  // CHECK-NEXT:    store <12 x i32> [[TRANS]], ptr %a_t, align 4

  int3x4 a_t = __builtin_matrix_transpose(a);
}

struct Foo {
  uint1x4 In;
  uint4x1 Out;
};

// CHECK-LABEL: define {{.*}}transpose_struct_member
void transpose_struct_member(struct Foo F) {
  // CHECK:         [[IN_PTR:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr %F, i32 0, i32 0
  // CHECK-NEXT:    [[M:%.*]] = load <4 x i32>, ptr [[IN_PTR]], align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <4 x i32> @llvm.matrix.transpose.v4i32(<4 x i32> [[M]], i32 1, i32 4)
  // CHECK-NEXT:    [[OUT_PTR:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr %F, i32 0, i32 1
  // CHECK-NEXT:    store <4 x i32> [[M_T]], ptr [[OUT_PTR]], align 4

  F.Out = __builtin_matrix_transpose(F.In);
}

// CHECK-LABEL: define {{.*}}transpose_transpose_struct_member
void transpose_transpose_struct_member(struct Foo F) {
  // CHECK:         [[IN_PTR:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr %F, i32 0, i32 0
  // CHECK-NEXT:    [[M:%.*]] = load <4 x i32>, ptr [[IN_PTR]], align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <4 x i32> @llvm.matrix.transpose.v4i32(<4 x i32> [[M]], i32 1, i32 4)
  // CHECK-NEXT:    [[M_T2:%.*]] = call <4 x i32> @llvm.matrix.transpose.v4i32(<4 x i32> [[M_T]], i32 4, i32 1)
  // CHECK:         [[OUT_PTR:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr %F, i32 0, i32 0
  // CHECK-NEXT:    store <4 x i32> [[M_T2]], ptr [[OUT_PTR]], align 4

  F.In = __builtin_matrix_transpose(__builtin_matrix_transpose(F.In));
}

double4x4 get_matrix(void);

// CHECK-LABEL: define {{.*}}transpose_rvalue
void transpose_rvalue(void) {
  // CHECK:         [[M_T_ADDR:%.*]] = alloca [16 x double], align 8
  // CHECK-NEXT:    [[CALL:%.*]] = call{{.*}} <16 x double> @_Z10get_matrixv()
  // CHECK-NEXT:    [[M_T:%.*]] = call <16 x double> @llvm.matrix.transpose.v16f64(<16 x double> [[CALL]], i32 4, i32 4)
  // CHECK-NEXT:   store <16 x double> [[M_T]], ptr [[M_T_ADDR]], align 8

  double4x4 m_t = __builtin_matrix_transpose(get_matrix());
}

double4x4 global_matrix;

// CHECK-LABEL: define {{.*}}transpose_global
void transpose_global(void) {
  // CHECK:         [[M_T_ADDR:%.*]] = alloca [16 x double], align 8
  // CHECK-NEXT:    [[GLOBAL_MATRIX:%.*]] = load <16 x double>, ptr @global_matrix, align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <16 x double> @llvm.matrix.transpose.v16f64(<16 x double> [[GLOBAL_MATRIX]], i32 4, i32 4)
  // CHECK-NEXT:    store <16 x double> [[M_T]], ptr [[M_T_ADDR]], align 8

  double4x4 m_t = __builtin_matrix_transpose(global_matrix);
}
