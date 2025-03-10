// RUN: %clang_cc1 -no-enable-noundef-analysis -triple spirv-unknown-vulkan-compute -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple dxil-pc-shadermodel6.3-compute -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Test the matrix type transpose builtin.

template <typename EltTy, unsigned Rows, unsigned Columns>
using matrix_t = matrix<EltTy, Rows, Columns>;

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  matrix_t<EltTy, Rows, Columns> value;
};

// Can't test utility function with matrix param without mangling.
template <typename T, unsigned R, unsigned C>
MyMatrix<T, C, R> transpose(const MyMatrix<T, R, C> M) {
  MyMatrix<T, C, R> Res;
  Res.value = __builtin_matrix_transpose(M.value);
  return Res;
}

// CHECK-LABEL: define{{.*}} void @_Z24test_transpose_template1v()
void test_transpose_template1() {
  // CHECK:         call{{.*}} void @_Z9transposeIiLj3ELj4EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.0) align 4 %M1_t, ptr byval(%struct.MyMatrix) align 4 %agg.tmp)
  // CHECK-LABEL: define{{.*}} void @_Z9transposeIiLj3ELj4EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <12 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <12 x i32> @llvm.matrix.transpose.v12i32(<12 x i32> [[M]], i32 3, i32 4)

  MyMatrix<int, 3, 4> M1;
  MyMatrix<int, 4, 3> M1_t = transpose(M1);
}

// CHECK-LABEL: define{{.*}} void @_Z24test_transpose_template2
void test_transpose_template2(inout MyMatrix<double, 3, 2> M) {
  // CHECK:         call{{.*}} void @_Z9transposeIdLj3ELj2EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.1) align 8 %agg.tmp1, ptr byval(%struct.MyMatrix.2) align 8 %agg.tmp2)
  // CHECK-NEXT:    call{{.*}} void @_Z9transposeIdLj2ELj3EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.2) align 8 %agg.tmp, ptr byval(%struct.MyMatrix.1) align 8 %agg.tmp1)
  // CHECK-NEXT:    call{{.*}} void @_Z9transposeIdLj3ELj2EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.1) align 8 %M2_t, ptr byval(%struct.MyMatrix.2) align 8 %agg.tmp)

  // CHECK-LABEL: define{{.*}} void @_Z9transposeIdLj3ELj2EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <6 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[M]], i32 3, i32 2)
  // CHECK-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds nuw %struct.MyMatrix.1, ptr %agg.result, i32 0, i32 0
  // CHECK-NEXT:    store <6 x double> [[M_T]], ptr [[RES_ADDR]], align 8

  // CHECK-LABEL: define{{.*}} void @_Z9transposeIdLj2ELj3EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <6 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[M]], i32 2, i32 3)
  // CHECK-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds nuw %struct.MyMatrix.2, ptr %agg.result, i32 0, i32 0
  // CHECK-NEXT:    store <6 x double> [[M_T]], ptr [[RES_ADDR]], align 8

  MyMatrix<double, 2, 3> M2_t = transpose(transpose(transpose(M)));
}

matrix_t<float, 3, 3> get_matrix();

// CHECK-LABEL: define{{.*}} void @_Z21test_transpose_rvaluev()
void test_transpose_rvalue() {
  // CHECK:         [[M_T_ADDR:%.*]] = alloca [9 x float], align 4
  // CHECK-NEXT:    [[CALL_RES:%.*]] = call{{.*}} <9 x float> @_Z10get_matrixv()
  // CHECK-NEXT:    [[ADD:%.*]] = fadd <9 x float> [[CALL_RES]], splat (float 2.000000e+00)
  // CHECK-NEXT:    [[M_T:%.*]] = call <9 x float> @llvm.matrix.transpose.v9f32(<9 x float> [[ADD]], i32 3, i32 3)
  // CHECK-NEXT:    store <9 x float> [[M_T]], ptr [[M_T_ADDR]], align 4
  matrix_t<float, 3, 3> m_t = __builtin_matrix_transpose(get_matrix() + 2.0);
}

// CHECK-LABEL:  define{{.*}} void @_Z20test_transpose_const
void test_transpose_const(const matrix_t<float, 3, 3> m) {
  // CHECK:         [[MATRIX:%.*]] = load <9 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <9 x float> @llvm.matrix.transpose.v9f32(<9 x float> [[MATRIX]], i32 3, i32 3)
  // CHECK-NEXT:    store <9 x float> [[M_T]], ptr %m_t, align 4
  matrix_t<float, 3, 3> m_t = __builtin_matrix_transpose(m);
}

// TODO: Enable once initialization support is defined and implemented for
//       matrix types.
// void test_lvalue_conversion() {
//  constexpr double4x4 m = {};
//  [] { return __builtin_matrix_transpose(m); }
//}

