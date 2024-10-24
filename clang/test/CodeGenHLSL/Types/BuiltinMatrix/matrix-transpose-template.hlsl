// RUN: %clang_cc1 -no-enable-noundef-analysis -triple spirv-unknown-vulkan-compute -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Test the matrix type transpose builtin.
// Since all these cases require mangling, DXIL is not tested for now.

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

void test_transpose_template1() {
  // SPIRV-LABEL: define{{.*}} void @_Z24test_transpose_template1v()
  // SPIRV:         call{{.*}} void @_Z9transposeIiLj3ELj4EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.0) align 4 %M1_t, ptr byval(%struct.MyMatrix) align 4 %agg.tmp)
  // SPIRV-LABEL: define{{.*}} void @_Z9transposeIiLj3ELj4EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(
  // SPIRV:         [[M:%.*]] = load <12 x i32>, ptr {{.*}}, align 4
  // SPIRV-NEXT:    [[M_T:%.*]] = call <12 x i32> @llvm.matrix.transpose.v12i32(<12 x i32> [[M]], i32 3, i32 4)

  MyMatrix<int, 3, 4> M1;
  MyMatrix<int, 4, 3> M1_t = transpose(M1);
}

void test_transpose_template2(inout MyMatrix<double, 3, 2> M) {
  // SPIRV-LABEL: define{{.*}} void @_Z24test_transpose_template2R8MyMatrixIdLj3ELj2EE(
  // SPIRV:         call{{.*}} void @_Z9transposeIdLj3ELj2EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.1) align 8 %agg.tmp1, ptr byval(%struct.MyMatrix.2) align 8 %agg.tmp2)
  // SPIRV-NEXT:    call{{.*}} void @_Z9transposeIdLj2ELj3EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.2) align 8 %agg.tmp, ptr byval(%struct.MyMatrix.1) align 8 %agg.tmp1)
  // SPIRV-NEXT:    call{{.*}} void @_Z9transposeIdLj3ELj2EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.1) align 8 %M2_t, ptr byval(%struct.MyMatrix.2) align 8 %agg.tmp)

  // SPIRV-LABEL: define{{.*}} void @_Z9transposeIdLj3ELj2EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(
  // SPIRV:         [[M:%.*]] = load <6 x double>, ptr {{.*}}, align 8
  // SPIRV-NEXT:    [[M_T:%.*]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[M]], i32 3, i32 2)
  // SPIRV-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds nuw %struct.MyMatrix.1, ptr %agg.result, i32 0, i32 0
  // SPIRV-NEXT:    store <6 x double> [[M_T]], ptr [[RES_ADDR]], align 8

  // SPIRV-LABEL: define{{.*}} void @_Z9transposeIdLj2ELj3EE8MyMatrixIT_XT1_EXT0_EES0_IS1_XT0_EXT1_EE(
  // SPIRV:         [[M:%.*]] = load <6 x double>, ptr {{.*}}, align 8
  // SPIRV-NEXT:    [[M_T:%.*]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[M]], i32 2, i32 3)
  // SPIRV-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds nuw %struct.MyMatrix.2, ptr %agg.result, i32 0, i32 0
  // SPIRV-NEXT:    store <6 x double> [[M_T]], ptr [[RES_ADDR]], align 8

  MyMatrix<double, 2, 3> M2_t = transpose(transpose(transpose(M)));
}

matrix_t<float, 3, 3> get_matrix();

void test_transpose_rvalue() {
  // CHECK-LABEL: define{{.*}} void @_Z21test_transpose_rvaluev()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    [[M_T_ADDR:%.*]] = alloca [9 x float], align 4
  // CHECK-NEXT:    [[CALL_RES:%.*]] = call{{.*}} <9 x float> @_Z10get_matrixv()
  // CHECK-NEXT:    [[ADD:%.*]] = fadd <9 x float> [[CALL_RES]], <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
  // CHECK-NEXT:    [[M_T:%.*]] = call <9 x float> @llvm.matrix.transpose.v9f32(<9 x float> [[ADD]], i32 3, i32 3)
  // CHECK-NEXT:    store <9 x float> [[M_T]], ptr [[M_T_ADDR]], align 4
  matrix_t<float, 3, 3> m_t = __builtin_matrix_transpose(get_matrix() + 2.0);
}

void test_transpose_const(const matrix_t<float, 3, 3> m) {
  // CHECK-LABEL:  define{{.*}} void @_Z20test_transpose_constu11matrix_typeILj3ELj3EfE(
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

