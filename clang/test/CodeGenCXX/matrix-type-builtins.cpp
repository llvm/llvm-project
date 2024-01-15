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
  // CHECK-LABEL: define{{.*}} void @_Z24test_transpose_template1v()
  // CHECK:         call void @_Z9transposeIiLj4ELj10EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.0) align 4 %M1_t, ptr noundef nonnull align 4 dereferenceable(160) %M1)

  // CHECK-LABEL: define linkonce_odr void @_Z9transposeIiLj4ELj10EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <40 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <40 x i32> @llvm.matrix.transpose.v40i32(<40 x i32> [[M]], i32 4, i32 10)

  MyMatrix<int, 4, 10> M1;
  MyMatrix<int, 10, 4> M1_t = transpose(M1);
}

void test_transpose_template2(MyMatrix<double, 7, 6> &M) {
  // CHECK-LABEL: define{{.*}} void @_Z24test_transpose_template2R8MyMatrixIdLj7ELj6EE(
  // CHECK:         call void @_Z9transposeIdLj7ELj6EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.1) align 8 %ref.tmp1, ptr noundef nonnull align 8 dereferenceable(336) %0)
  // CHECK-NEXT:    call void @_Z9transposeIdLj6ELj7EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.2) align 8 %ref.tmp, ptr noundef nonnull align 8 dereferenceable(336) %ref.tmp1)
  // CHECK-NEXT:    call void @_Z9transposeIdLj7ELj6EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(ptr dead_on_unwind writable sret(%struct.MyMatrix.1) align 8 %M2_t, ptr noundef nonnull align 8 dereferenceable(336) %ref.tmp)

  // CHECK-LABEL: define linkonce_odr void @_Z9transposeIdLj7ELj6EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <42 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <42 x double> @llvm.matrix.transpose.v42f64(<42 x double> [[M]], i32 7, i32 6)
  // CHECK-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds %struct.MyMatrix.1, ptr %agg.result, i32 0, i32 0
  // CHECK-NEXT:    store <42 x double> [[M_T]], ptr [[RES_ADDR]], align 8

  // CHECK-LABEL: define linkonce_odr void @_Z9transposeIdLj6ELj7EE8MyMatrixIT_XT1_EXT0_EERKS0_IS1_XT0_EXT1_EE(
  // CHECK:         [[M:%.*]] = load <42 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <42 x double> @llvm.matrix.transpose.v42f64(<42 x double> [[M]], i32 6, i32 7)
  // CHECK-NEXT:    [[RES_ADDR:%.*]] = getelementptr inbounds %struct.MyMatrix.2, ptr %agg.result, i32 0, i32 0
  // CHECK-NEXT:    store <42 x double> [[M_T]], ptr [[RES_ADDR]], align 8

  MyMatrix<double, 6, 7> M2_t = transpose(transpose(transpose(M)));
}

matrix_t<float, 3, 3> get_matrix();

void test_transpose_rvalue() {
  // CHECK-LABEL: define{{.*}} void @_Z21test_transpose_rvaluev()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[M_T_ADDR:%.*]] = alloca [9 x float], align 4
  // CHECK-NEXT:    [[CALL_RES:%.*]] = call noundef <9 x float> @_Z10get_matrixv()
  // CHECK-NEXT:    [[ADD:%.*]] = fadd <9 x float> [[CALL_RES]], <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
  // CHECK-NEXT:    [[M_T:%.*]] = call <9 x float> @llvm.matrix.transpose.v9f32(<9 x float> [[ADD]], i32 3, i32 3)
  // CHECK-NEXT:    store <9 x float> [[M_T]], ptr [[M_T_ADDR]], align 4
  matrix_t<float, 3, 3> m_t = __builtin_matrix_transpose(get_matrix() + 2.0);
}

void test_transpose_const(const matrix_t<float, 3, 3> &m) {
  // CHECK-LABEL:  define{{.*}} void @_Z20test_transpose_constRKu11matrix_typeILm3ELm3EfE(
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

template <typename T, unsigned R, unsigned C, unsigned S>
matrix_t<T, R, C> column_major_load_with_stride(T *Ptr) {
  return __builtin_matrix_column_major_load(Ptr, R, C, S);
}

void test_column_major_load_with_stride_template_double(double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z50test_column_major_load_with_stride_template_doublePd(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call noundef <40 x double> @_Z29column_major_load_with_strideIdLj10ELj4ELj15EEu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef [[PTR]])

  // CHECK-LABEL:  define linkonce_odr noundef <40 x double> @_Z29column_major_load_with_strideIdLj10ELj4ELj15EEu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call <40 x double> @llvm.matrix.column.major.load.v40f64.i64(ptr align 8 [[PTR]], i64 15, i1 false, i32 10, i32 4)

  matrix_t<double, 10, 4> M1 = column_major_load_with_stride<double, 10, 4, 15>(Ptr);
}

void test_column_major_load_with_stride_template_int(int *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z47test_column_major_load_with_stride_template_intPi(ptr noundef %Ptr) #5 {
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call noundef <6 x i32> @_Z29column_major_load_with_strideIiLj3ELj2ELj12EEu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef [[PTR]])

  // CHECK-LABEL: define linkonce_odr noundef <6 x i32> @_Z29column_major_load_with_strideIiLj3ELj2ELj12EEu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call <6 x i32> @llvm.matrix.column.major.load.v6i32.i64(ptr align 4 [[PTR]], i64 12, i1 false, i32 3, i32 2)

  matrix_t<int, 3, 2> M1 = column_major_load_with_stride<int, 3, 2, 12>(Ptr);
}

struct UnsignedWrapper {
  char x;
  operator unsigned() {
    return x;
  }
};

void test_column_major_load_stride_wrapper(int *Ptr, UnsignedWrapper &W) {
  // CHECK-LABEL:  define{{.*}} void @_Z37test_column_major_load_stride_wrapperPiR15UnsignedWrapper(ptr noundef %Ptr, ptr noundef nonnull align 1 dereferenceable(1) %W)
  // CHECK:         [[W:%.*]] = load ptr, ptr %W.addr, align 8
  // CHECK-NEXT:    [[STRIDE:%.*]] = call noundef i32 @_ZN15UnsignedWrappercvjEv(ptr {{[^,]*}} [[W]])
  // CHECK-NEXT:    [[STRIDE_EXT:%.*]] = zext i32 [[STRIDE]] to i64
  // CHECK-NEXT:    [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call <4 x i32> @llvm.matrix.column.major.load.v4i32.i64(ptr align 4 [[PTR]], i64 [[STRIDE_EXT]], i1 false, i32 2, i32 2)
  matrix_t<int, 2, 2> M1 = __builtin_matrix_column_major_load(Ptr, 2, 2, W);
}

constexpr int constexpr3() { return 3; }

void test_column_major_load_constexpr_num_rows(int *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z41test_column_major_load_constexpr_num_rowsPi(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call <6 x i32> @llvm.matrix.column.major.load.v6i32.i64(ptr align 4 [[PTR]], i64 3, i1 false, i32 3, i32 2)

  matrix_t<int, 3, 2> M1 = __builtin_matrix_column_major_load(Ptr, constexpr3(), 2, 3);
}

constexpr int constexpr1() { return 1; }

void test_column_major_load_constexpr_num_columns(int *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z44test_column_major_load_constexpr_num_columnsPi(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call <2 x i32> @llvm.matrix.column.major.load.v2i32.i64(ptr align 4 [[PTR]], i64 3, i1 false, i32 2, i32 1)
  matrix_t<int, 2, 1> M1 = __builtin_matrix_column_major_load(Ptr, 2, constexpr1(), 3);
}

template <unsigned N>
constexpr int constexpr_plus1() { return N + 1; }

void test_column_major_load_constexpr_num_columns_temp(int *Ptr) {
  // CHECK-LABEL:  define{{.*}} void @_Z49test_column_major_load_constexpr_num_columns_tempPi(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call <10 x i32> @llvm.matrix.column.major.load.v10i32.i64(ptr align 4 [[PTR]], i64 3, i1 false, i32 2, i32 5)
  matrix_t<int, 2, 5> M1 = __builtin_matrix_column_major_load(Ptr, 2, constexpr_plus1<4>(), 3);
}

void test_column_major_load_constexpr_stride_constexpr(int *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z49test_column_major_load_constexpr_stride_constexprPi(ptr noundef %Ptr)
  // CHECK:         [[STRIDE:%.*]] = call noundef i32 @_Z10constexpr3v()
  // CHECK-NEXT:    [[STRIDE_EXT:%.*]] = sext i32 [[STRIDE]] to i64
  // CHECK-NEXT:    [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call <4 x i32> @llvm.matrix.column.major.load.v4i32.i64(ptr align 4 [[PTR]], i64 [[STRIDE_EXT]], i1 false, i32 2, i32 2)

  matrix_t<int, 2, 2> M1 = __builtin_matrix_column_major_load(Ptr, 2, 2, constexpr3());
}

template <typename T>
struct remove_pointer {
  typedef T type;
};

template <typename T>
struct remove_pointer<T *> {
  typedef typename remove_pointer<T>::type type;
};

// Same as column_major_load_with_stride, but with the PtrT argument itself begin a pointer type.
template <typename PtrT, unsigned R, unsigned C, unsigned S>
matrix_t<typename remove_pointer<PtrT>::type, R, C> column_major_load_with_stride2(PtrT Ptr) {
  return __builtin_matrix_column_major_load(Ptr, R, C, S);
}

void call_column_major_load_with_stride2(float *Ptr) {
  matrix_t<float, 2, 2> m = column_major_load_with_stride2<float *, 2, 2, 2>(Ptr);
}

template <typename T, unsigned R, unsigned C, unsigned S>
void column_major_store_with_stride(matrix_t<T, R, C> &m, T *Ptr) {
  __builtin_matrix_column_major_store(m, Ptr, S);
}

void test_column_major_store_with_stride_template_double(double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z51test_column_major_store_with_stride_template_doublePd(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call void @_Z30column_major_store_with_strideIdLj10ELj4ELj15EEvRu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef nonnull align 8 dereferenceable(320) %M1, ptr noundef [[PTR]])

  // CHECK-LABEL:  define linkonce_odr void @_Z30column_major_store_with_strideIdLj10ELj4ELj15EEvRu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef nonnull align 8 dereferenceable(320) %m, ptr noundef %Ptr)
  // CHECK:         [[M:%.*]] = load <40 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:    [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v40f64.i64(<40 x double> [[M]], ptr align 8 [[PTR]], i64 15, i1 false, i32 10, i32 4)

  matrix_t<double, 10, 4> M1;
  column_major_store_with_stride<double, 10, 4, 15>(M1, Ptr);
}

void test_column_major_store_with_stride_template_int(int *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z48test_column_major_store_with_stride_template_intPi(ptr noundef %Ptr)
  // CHECK:         [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call void @_Z30column_major_store_with_strideIiLj3ELj2ELj3EEvRu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef nonnull align 4 dereferenceable(24) %M1, ptr noundef [[PTR]])

  // CHECK-LABEL:  define linkonce_odr void @_Z30column_major_store_with_strideIiLj3ELj2ELj3EEvRu11matrix_typeIXT0_EXT1_ET_EPS0_(ptr noundef nonnull align 4 dereferenceable(24) %m, ptr noundef %Ptr)
  // CHECK:         [[M:%.*]] = load <6 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:    [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v6i32.i64(<6 x i32> [[M]], ptr align 4 [[PTR]], i64 3, i1 false, i32 3, i32 2)

  matrix_t<int, 3, 2> M1;
  column_major_store_with_stride<int, 3, 2, 3>(M1, Ptr);
}

void test_column_major_store_stride_wrapper(int *Ptr, UnsignedWrapper &W) {
  // CHECK-LABEL: define{{.*}} void @_Z38test_column_major_store_stride_wrapperPiR15UnsignedWrapper(ptr noundef %Ptr, ptr noundef nonnull align 1 dereferenceable(1) %W)
  // CHECK:         [[M:%.*]] = load <4 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:    [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    [[W:%.*]] = load ptr, ptr %W.addr, align 8
  // CHECK-NEXT:    [[IDX:%.*]] = call noundef i32 @_ZN15UnsignedWrappercvjEv(ptr {{[^,]*}} [[W]])
  // CHECK-NEXT:    [[IDX_EXT:%.*]] = zext i32 [[IDX]] to i64
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v4i32.i64(<4 x i32> [[M]], ptr align 4 [[PTR]], i64 [[IDX_EXT]], i1 false, i32 2, i32 2)

  matrix_t<int, 2, 2> M1;
  __builtin_matrix_column_major_store(M1, Ptr, W);
}

void test_column_major_store_constexpr_stride_constexpr(int *Ptr) {
  // CHECK-LABEL: define{{.*}} void @_Z50test_column_major_store_constexpr_stride_constexprPi(ptr noundef %Ptr)
  // CHECK:         [[M:%.*]] = load <4 x i32>, ptr %M, align 4
  // CHECK-NEXT:    [[PTR:%.*]] = load ptr, ptr %Ptr.addr, align 8
  // CHECK-NEXT:    [[IDX:%.*]] = call noundef i32 @_Z10constexpr3v()
  // CHECK-NEXT:    [[IDX_EXT:%.*]] = sext i32 [[IDX]] to i64
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v4i32.i64(<4 x i32> [[M]], ptr align 4 [[PTR]], i64 [[IDX_EXT]], i1 false, i32 2, i32 2)

  matrix_t<int, 2, 2> M;
  __builtin_matrix_column_major_store(M, Ptr, constexpr3());
}
