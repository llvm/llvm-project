// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - -std=c++17 | FileCheck %s

// Tests for the matrix type builtins.

template <typename EltTy, unsigned Rows, unsigned Columns>
using matrix_t = EltTy __attribute__((matrix_type(Rows, Columns)));

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  matrix_t<EltTy, Rows, Columns> value;
};

template <typename T, unsigned R, unsigned C>
MyMatrix<T, C, R> transpose(const MyMatrix<T, R, C> &M) {
  MyMatrix<T, C, R> Res;
  Res.value = __builtin_matrix_transpose(M.value);
  return Res;
}

void test_transpose_template1() {
  // CHECK-LABEL: define void @_Z24test_transpose_template1v()
  // CHECK:         call void @_Z9transposeIiLj4ELj10EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(%struct.MyMatrix.0* sret align 4 %M1_t, %struct.MyMatrix* nonnull align 4 dereferenceable(160) %M1)

  // CHECK-LABEL: define linkonce_odr void @_Z9transposeIiLj4ELj10EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <40 x i32>, <40 x i32>* {{.*}}, align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <40 x i32> @llvm.matrix.transpose.v40i32(<40 x i32> [[M]], i32 4, i32 10)

  MyMatrix<int, 4, 10> M1;
  MyMatrix<int, 10, 4> M1_t = transpose(M1);
}

void test_transpose_template2(MyMatrix<double, 7, 6> &M) {
  // CHECK-LABEL: define void @_Z24test_transpose_template2R8MyMatrixIdLj7ELj6EE(
  // CHECK:         call void @_Z9transposeIdLj7ELj6EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(%struct.MyMatrix.2* sret align 8 %ref.tmp1, %struct.MyMatrix.1* nonnull align 8 dereferenceable(336) %0)
  // CHECK-NEXT:    call void @_Z9transposeIdLj6ELj7EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(%struct.MyMatrix.1* sret align 8 %ref.tmp, %struct.MyMatrix.2* nonnull align 8 dereferenceable(336) %ref.tmp1)
  // CHECK-NEXT:    call void @_Z9transposeIdLj7ELj6EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(%struct.MyMatrix.2* sret align 8 %M2_t, %struct.MyMatrix.1* nonnull align 8 dereferenceable(336) %ref.tmp)

  // CHECK-LABEL: define linkonce_odr void @_Z9transposeIdLj7ELj6EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <42 x double>, <42 x double>* {{.*}}, align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <42 x double> @llvm.matrix.transpose.v42f64(<42 x double> [[M]], i32 7, i32 6)
  // CHECK-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds %struct.MyMatrix.2, %struct.MyMatrix.2* %agg.result, i32 0, i32 0
  // CHECK-NEXT:    [[RES_ADDR_C:%.*]] = bitcast [42 x double]* [[RES_ADDR]] to <42 x double>*
  // CHECK-NEXT:    store <42 x double> [[M_T]], <42 x double>* [[RES_ADDR_C]], align 8

  // CHECK-LABEL: define linkonce_odr void @_Z9transposeIdLj6ELj7EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <42 x double>, <42 x double>* {{.*}}, align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <42 x double> @llvm.matrix.transpose.v42f64(<42 x double> [[M]], i32 6, i32 7)
  // CHECK-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds %struct.MyMatrix.1, %struct.MyMatrix.1* %agg.result, i32 0, i32 0
  // CHECK-NEXT:    [[RES_ADDR_C:%.*]] = bitcast [42 x double]* [[RES_ADDR]] to <42 x double>*
  // CHECK-NEXT:    store <42 x double> [[M_T]], <42 x double>* [[RES_ADDR_C]], align 8

  MyMatrix<double, 6, 7> M2_t = transpose(transpose(transpose(M)));
}

matrix_t<float, 3, 3> get_matrix();

void test_transpose_rvalue() {
  // CHECK-LABEL: define void @_Z21test_transpose_rvaluev()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[M_T_ADDR:%.*]] = alloca [9 x float], align 4
  // CHECK-NEXT:    [[CALL_RES:%.*]] = call <9 x float> @_Z10get_matrixv()
  // CHECK-NEXT:    [[ADD:%.*]] = fadd <9 x float> [[CALL_RES]], <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
  // CHECK-NEXT:    [[M_T:%.*]] = call <9 x float> @llvm.matrix.transpose.v9f32(<9 x float> [[ADD]], i32 3, i32 3)
  // CHECK-NEXT:    [[M_T_ADDR_CAST:%.*]] = bitcast [9 x float]* [[M_T_ADDR]] to <9 x float>*
  // CHECK-NEXT:    store <9 x float> [[M_T]], <9 x float>* [[M_T_ADDR_CAST]], align 4
  matrix_t<float, 3, 3> m_t = __builtin_matrix_transpose(get_matrix() + 2.0);
}

void test_transpose_const(const matrix_t<float, 3, 3> &m) {
  // CHECK-LABEL:  define void @_Z20test_transpose_constRKU11matrix_typeLm3ELm3Ef(
  // CHECK:         [[MATRIX:%.*]] = load <9 x float>, <9 x float>* {{.*}}, align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <9 x float> @llvm.matrix.transpose.v9f32(<9 x float> [[MATRIX]], i32 3, i32 3)
  // CHECK-NEXT:    [[M_T_ADDR:%.*]] = bitcast [9 x float]* %m_t to <9 x float>*
  // CHECK-NEXT:    store <9 x float> [[M_T]], <9 x float>* [[M_T_ADDR]], align 4
  matrix_t<float, 3, 3> m_t = __builtin_matrix_transpose(m);
}

// TODO: Enable once initialization support is defined and implemented for
//       matrix types.
// void test_lvalue_conversion() {
//  constexpr double4x4 m = {};
//  [] { return __builtin_matrix_transpose(m); }
//}
