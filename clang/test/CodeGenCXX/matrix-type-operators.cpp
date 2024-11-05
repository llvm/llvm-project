// RUN: %clang_cc1 -O0 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - -std=c++11 | FileCheck --check-prefixes=CHECK,NOOPT %s
// RUN: %clang_cc1 -O1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - -std=c++11 | FileCheck --check-prefixes=CHECK,OPT %s
typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
using fx2x3_t = float __attribute__((matrix_type(2, 3)));

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  using matrix_t = EltTy __attribute__((matrix_type(Rows, Columns)));

  matrix_t value;
};

template <typename EltTy0, unsigned R0, unsigned C0>
typename MyMatrix<EltTy0, R0, C0>::matrix_t add(MyMatrix<EltTy0, R0, C0> &A, MyMatrix<EltTy0, R0, C0> &B) {
  return A.value + B.value;
}

void test_add_template() {
  // CHECK-LABEL: define{{.*}} void @_Z17test_add_templatev()
  // CHECK:       %call = call noundef <10 x float> @_Z3addIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(ptr noundef nonnull align 4 dereferenceable(40) %Mat1, ptr noundef nonnull align 4 dereferenceable(40) %Mat2)

  // CHECK-LABEL: define linkonce_odr noundef <10 x float> @_Z3addIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(
  // NOOPT:       [[MAT1:%.*]] = load <10 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT:       [[MAT2:%.*]] = load <10 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[MAT1:%.*]] = load <10 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:         [[MAT2:%.*]] = load <10 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fadd <10 x float> [[MAT1]], [[MAT2]]
  // CHECK-NEXT:  ret <10 x float> [[RES]]

  MyMatrix<float, 2, 5> Mat1;
  MyMatrix<float, 2, 5> Mat2;
  Mat1.value = add(Mat1, Mat2);
}

template <typename EltTy0, unsigned R0, unsigned C0>
typename MyMatrix<EltTy0, R0, C0>::matrix_t subtract(MyMatrix<EltTy0, R0, C0> &A, MyMatrix<EltTy0, R0, C0> &B) {
  return A.value - B.value;
}

void test_subtract_template() {
  // CHECK-LABEL: define{{.*}} void @_Z22test_subtract_templatev()
  // CHECK:       %call = call noundef <10 x float> @_Z8subtractIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(ptr noundef nonnull align 4 dereferenceable(40) %Mat1, ptr noundef nonnull align 4 dereferenceable(40) %Mat2)

  // CHECK-LABEL: define linkonce_odr noundef <10 x float> @_Z8subtractIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(
  // NOOPT:       [[MAT1:%.*]] = load <10 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT:       [[MAT2:%.*]] = load <10 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[MAT1:%.*]] = load <10 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:         [[MAT2:%.*]] = load <10 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fsub <10 x float> [[MAT1]], [[MAT2]]
  // CHECK-NEXT:  ret <10 x float> [[RES]]

  MyMatrix<float, 2, 5> Mat1;
  MyMatrix<float, 2, 5> Mat2;
  Mat1.value = subtract(Mat1, Mat2);
}

struct DoubleWrapper1 {
  int x;
  operator double() {
    return x;
  }
};

void test_DoubleWrapper1_Sub1(MyMatrix<double, 10, 9> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z24test_DoubleWrapper1_Sub1R8MyMatrixIdLj10ELj9EE(
  // NOOPT:       [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR:%.*]] = call noundef double @_ZN14DoubleWrapper1cvdEv(ptr {{[^,]*}} %w1)
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <90 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <90 x double> [[SCALAR_EMBED]], <90 x double> poison, <90 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <90 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK:       store <90 x double> [[RES]], ptr {{.*}}, align 8

  DoubleWrapper1 w1;
  w1.x = 10;
  m.value = m.value - w1;
}

void test_DoubleWrapper1_Sub2(MyMatrix<double, 10, 9> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z24test_DoubleWrapper1_Sub2R8MyMatrixIdLj10ELj9EE(
  // CHECK:       [[SCALAR:%.*]] = call noundef double @_ZN14DoubleWrapper1cvdEv(ptr {{[^,]*}} %w1)
  // NOOPT:       [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <90 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <90 x double> [[SCALAR_EMBED]], <90 x double> poison, <90 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <90 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <90 x double> [[RES]], ptr {{.*}}, align 8

  DoubleWrapper1 w1;
  w1.x = 10;
  m.value = w1 - m.value;
}

struct DoubleWrapper2 {
  int x;
  operator double() {
    return x;
  }
};

void test_DoubleWrapper2_Add1(MyMatrix<double, 10, 9> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z24test_DoubleWrapper2_Add1R8MyMatrixIdLj10ELj9EE(
  // NOOPT:       [[MATRIX:%.*]] = load <90 x double>, ptr {{.+}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <90 x double>, ptr {{.+}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK:       [[SCALAR:%.*]] = call noundef double @_ZN14DoubleWrapper2cvdEv(ptr {{[^,]*}} %w2)
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <90 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <90 x double> [[SCALAR_EMBED]], <90 x double> poison, <90 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <90 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK:       store <90 x double> [[RES]], ptr {{.*}}, align 8

  DoubleWrapper2 w2;
  w2.x = 20;
  m.value = m.value + w2;
}

void test_DoubleWrapper2_Add2(MyMatrix<double, 10, 9> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z24test_DoubleWrapper2_Add2R8MyMatrixIdLj10ELj9EE(
  // CHECK:       [[SCALAR:%.*]] = call noundef double @_ZN14DoubleWrapper2cvdEv(ptr {{[^,]*}} %w2)
  // NOOPT:       [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <90 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <90 x double> [[SCALAR_EMBED]], <90 x double> poison, <90 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <90 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <90 x double> [[RES]], ptr {{.*}}, align 8

  DoubleWrapper2 w2;
  w2.x = 20;
  m.value = w2 + m.value;
}

struct IntWrapper {
  char x;
  operator int() {
    return x;
  }
};

void test_IntWrapper_Add(MyMatrix<double, 10, 9> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z19test_IntWrapper_AddR8MyMatrixIdLj10ELj9EE(
  // NOOPT:       [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR:%.*]] = call noundef i32 @_ZN10IntWrappercviEv(ptr {{[^,]*}} %w3)
  // CHECK-NEXT:  [[SCALAR_FP:%.*]] = sitofp i32 %call to double
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <90 x double> poison, double [[SCALAR_FP]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <90 x double> [[SCALAR_EMBED]], <90 x double> poison, <90 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <90 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK:       store <90 x double> [[RES]], ptr {{.*}}, align 8

  IntWrapper w3;
  w3.x = 'c';
  m.value = m.value + w3;
}

void test_IntWrapper_Sub(MyMatrix<double, 10, 9> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z19test_IntWrapper_SubR8MyMatrixIdLj10ELj9EE(
  // CHECK:       [[SCALAR:%.*]] = call noundef i32 @_ZN10IntWrappercviEv(ptr {{[^,]*}} %w3)
  // CHECK-NEXT:  [[SCALAR_FP:%.*]] = sitofp i32 %call to double
  // NOOPT:       [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <90 x double> poison, double [[SCALAR_FP]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <90 x double> [[SCALAR_EMBED]], <90 x double> poison, <90 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <90 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <90 x double> [[RES]], ptr {{.*}}, align 8

  IntWrapper w3;
  w3.x = 'c';
  m.value = w3 - m.value;
}

template <typename EltTy0, unsigned R0, unsigned C0, unsigned C1>
typename MyMatrix<EltTy0, R0, C1>::matrix_t multiply(MyMatrix<EltTy0, R0, C0> &A, MyMatrix<EltTy0, C0, C1> &B) {
  return A.value * B.value;
}

MyMatrix<float, 2, 2> test_multiply_template(MyMatrix<float, 2, 5> Mat1,
                                             MyMatrix<float, 5, 2> Mat2) {
  // CHECK-LABEL: define{{.*}} void @_Z22test_multiply_template8MyMatrixIfLj2ELj5EES_IfLj5ELj2EE(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[RES:%.*]] = call noundef <4 x float> @_Z8multiplyIfLj2ELj5ELj2EEN8MyMatrixIT_XT0_EXT2_EE8matrix_tERS0_IS1_XT0_EXT1_EERS0_IS1_XT1_EXT2_EE(ptr noundef nonnull align 4 dereferenceable(40) %Mat1, ptr noundef nonnull align 4 dereferenceable(40) %Mat2)
  // CHECK-NEXT:    %value = getelementptr inbounds nuw %struct.MyMatrix.1, ptr %agg.result, i32 0, i32 0
  // CHECK-NEXT:    store <4 x float> [[RES]], ptr %value, align 4
  // CHECK-NEXT:    ret void
  //
  // CHECK-LABEL:  define linkonce_odr noundef <4 x float> @_Z8multiplyIfLj2ELj5ELj2EEN8MyMatrixIT_XT0_EXT2_EE8matrix_tERS0_IS1_XT0_EXT1_EERS0_IS1_XT1_EXT2_EE(
  // NOOPT:         [[MAT1:%.*]] = load <10 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT:         [[MAT2:%.*]] = load <10 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT1:%.*]] = load <10 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:           [[MAT2:%.*]] = load <10 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = call <4 x float> @llvm.matrix.multiply.v4f32.v10f32.v10f32(<10 x float> [[MAT1]], <10 x float> [[MAT2]], i32 2, i32 5, i32 2)
  // CHECK-NEXT:    ret <4 x float> [[RES]]

  MyMatrix<float, 2, 2> Res;
  Res.value = multiply(Mat1, Mat2);
  return Res;
}

void test_IntWrapper_Multiply(MyMatrix<double, 10, 9> &m, IntWrapper &w3) {
  // CHECK-LABEL: define{{.*}} void @_Z24test_IntWrapper_MultiplyR8MyMatrixIdLj10ELj9EER10IntWrapper(
  // CHECK:       [[SCALAR:%.*]] = call noundef i32 @_ZN10IntWrappercviEv(ptr noundef {{.*}})
  // CHECK-NEXT:  [[SCALAR_FP:%.*]] = sitofp i32 %call to double
  // NOOPT:       [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <90 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <90 x double> poison, double [[SCALAR_FP]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <90 x double> [[SCALAR_EMBED]], <90 x double> poison, <90 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fmul <90 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <90 x double> [[RES]], ptr {{.*}}, align 8
  // CHECK-NEXT:  ret void
  m.value = w3 * m.value;
}

template <typename EltTy, unsigned Rows, unsigned Columns>
void insert(MyMatrix<EltTy, Rows, Columns> &Mat, EltTy e, unsigned i, unsigned j) {
  Mat.value[i][j] = e;
}

void test_insert_template1(MyMatrix<unsigned, 2, 2> &Mat, unsigned e, unsigned i, unsigned j) {
  // CHECK-LABEL: @_Z21test_insert_template1R8MyMatrixIjLj2ELj2EEjjj(
  // NOOPT:         [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8{{$}}
  // NOOPT-NEXT:    [[E:%.*]] = load i32, ptr %e.addr, align 4{{$}}
  // NOOPT-NEXT:    [[I:%.*]] = load i32, ptr %i.addr, align 4{{$}}
  // NOOPT-NEXT:    [[J:%.*]] = load i32, ptr %j.addr, align 4{{$}}
  // OPT:           [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[E:%.*]] = load i32, ptr %e.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[I:%.*]] = load i32, ptr %i.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i32, ptr %j.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    call void @_Z6insertIjLj2ELj2EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noundef nonnull align 4 dereferenceable(16) [[MAT_ADDR]], i32 noundef [[E]], i32 noundef [[I]], i32 noundef [[J]])
  // CHECK-NEXT:    ret void
  //
  // CHECK-LABEL: define linkonce_odr void @_Z6insertIjLj2ELj2EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(
  // NOOPT:         [[E:%.*]] = load i32, ptr %e.addr, align 4{{$}}
  // NOOPT:         [[I:%.*]] = load i32, ptr %i.addr, align 4{{$}}
  // OPT:           [[E:%.*]] = load i32, ptr %e.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:           [[I:%.*]] = load i32, ptr %i.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[I_EXT:%.*]] = zext i32 [[I]] to i64
  // NOOPT-NEXT:    [[J:%.*]] = load i32, ptr %j.addr, align 4{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i32, ptr %j.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[J_EXT:%.*]] = zext i32 [[J]] to i64
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[J_EXT]], 2
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], [[I_EXT]]
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 4
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT:%.*]] = load <4 x i32>, ptr {{.*}}, align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <4 x i32> [[MAT]], i32 [[E]], i64 [[IDX2]]
  // CHECK-NEXT:    store <4 x i32> [[MATINS]], ptr {{.*}}, align 4
  // CHECK-NEXT:    ret void

  insert(Mat, e, i, j);
}

void test_insert_template2(MyMatrix<float, 3, 8> &Mat, float e) {
  // CHECK-LABEL: @_Z21test_insert_template2R8MyMatrixIfLj3ELj8EEf(
  // NOOPT:         [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8{{$}}
  // NOOPT-NEXT:    [[E:%.*]] = load float, ptr %e.addr, align 4{{$}}
  // OPT:           [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[E:%.*]] = load float, ptr %e.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    call void @_Z6insertIfLj3ELj8EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noundef nonnull align 4 dereferenceable(96) [[MAT_ADDR]], float noundef [[E]], i32 noundef 2, i32 noundef 5)
  // CHECK-NEXT:    ret void
  //
  // CHECK-LABEL: define linkonce_odr void @_Z6insertIfLj3ELj8EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(
  // NOOPT:         [[E:%.*]] = load float, ptr %e.addr, align 4{{$}}
  // NOOPT:         [[I:%.*]] = load i32, ptr %i.addr, align 4{{$}}
  // OPT:           [[E:%.*]] = load float, ptr %e.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:           [[I:%.*]] = load i32, ptr %i.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[I_EXT:%.*]] = zext i32 [[I]] to i64
  // NOOPT-NEXT:    [[J:%.*]] = load i32, ptr %j.addr, align 4{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i32, ptr %j.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[J_EXT:%.*]] = zext i32 [[J]] to i64
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[J_EXT]], 3
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], [[I_EXT]]
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 24
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT:%.*]] = load <24 x float>, ptr {{.*}}, align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <24 x float> [[MAT]], float [[E]], i64 [[IDX2]]
  // CHECK-NEXT:    store <24 x float> [[MATINS]], ptr {{.*}}, align 4
  // CHECK-NEXT:    ret void

  insert(Mat, e, 2, 5);
}

template <typename EltTy, unsigned Rows, unsigned Columns>
EltTy extract(MyMatrix<EltTy, Rows, Columns> &Mat) {
  return Mat.value[1u][0u];
}

int test_extract_template(MyMatrix<int, 2, 2> Mat1) {
  // CHECK-LABEL: @_Z21test_extract_template8MyMatrixIiLj2ELj2EE(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[CALL:%.*]] = call noundef i32 @_Z7extractIiLj2ELj2EET_R8MyMatrixIS0_XT0_EXT1_EE(ptr noundef nonnull align 4 dereferenceable(16) [[MAT1:%.*]])
  // CHECK-NEXT:    ret i32 [[CALL]]
  //
  // CHECK-LABEL: define linkonce_odr noundef i32 @_Z7extractIiLj2ELj2EET_R8MyMatrixIS0_XT0_EXT1_EE(
  // NOOPT:         [[MAT:%.*]] = load <4 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <4 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <4 x i32> [[MAT]], i64 1
  // CHECK-NEXT:    ret i32 [[MATEXT]]

  return extract(Mat1);
}

using double4x4 = double __attribute__((matrix_type(4, 4)));

template <class R, class C>
auto matrix_subscript(double4x4 m, R r, C c) -> decltype(m[r][c]) {}

double test_matrix_subscript(double4x4 m) {
  // CHECK-LABEL: @_Z21test_matrix_subscriptu11matrix_typeILm4ELm4EdE(
  // NOOPT:         [[MAT:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:           [[MAT:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[CALL:%.*]] = call noundef nonnull align 8 dereferenceable(8) ptr @_Z16matrix_subscriptIiiEDTixixfp_fp0_fp1_Eu11matrix_typeILm4ELm4EdET_T0_(<16 x double> noundef [[MAT]], i32 noundef 1, i32 noundef 2)
  // NOOPT-NEXT:    [[RES:%.*]] = load double, ptr [[CALL]], align 8{{$}}
  // OPT-NEXT:      [[RES:%.*]] = load double, ptr [[CALL]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    ret double [[RES]]

  return matrix_subscript(m, 1, 2);
}

const double &test_matrix_subscript_reference(const double4x4 m) {
  // CHECK-LABEL: @_Z31test_matrix_subscript_referenceu11matrix_typeILm4ELm4EdE(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[M_ADDR:%.*]] = alloca [16 x double], align 8
  // CHECK-NEXT:    [[REF_TMP:%.*]] = alloca double, align 8
  // CHECK-NEXT:    store <16 x double> [[M:%.*]], ptr [[M_ADDR]], align 8
  // NOOPT:         [[NAMELESS1:%.*]] = load <16 x double>, ptr [[M_ADDR]], align 8{{$}}
  // OPT:           [[NAMELESS1:%.*]] = load <16 x double>, ptr [[M_ADDR]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <16 x double> [[NAMELESS1]], i64 4
  // CHECK-NEXT:    store double [[MATEXT]], ptr [[REF_TMP]], align 8
  // CHECK:         ret ptr [[REF_TMP]]

  return m[0][1];
}

struct UnsignedWrapper {
  char x;
  operator unsigned() {
    return x;
  }
};

double extract_IntWrapper_idx(double4x4 &m, IntWrapper i, UnsignedWrapper j) {
  // CHECK-LABEL: define{{.*}} double @_Z22extract_IntWrapper_idxRu11matrix_typeILm4ELm4EdE10IntWrapper15UnsignedWrapper(
  // CHECK:         [[I:%.*]] = call noundef i32 @_ZN10IntWrappercviEv(ptr {{[^,]*}} %i)
  // CHECK-NEXT:    [[I_ADD:%.*]] = add nsw i32 [[I]], 1
  // CHECK-NEXT:    [[I_ADD_EXT:%.*]] = sext i32 [[I_ADD]] to i64
  // CHECK-NEXT:    [[J:%.*]] = call noundef i32 @_ZN15UnsignedWrappercvjEv(ptr {{[^,]*}} %j)
  // CHECK-NEXT:    [[J_SUB:%.*]] = sub i32 [[J]], 1
  // CHECK-NEXT:    [[J_SUB_EXT:%.*]] = zext i32 [[J_SUB]] to i64
  // CHECK-NEXT:    [[IDX1:%.*]] = mul i64 [[J_SUB_EXT]], 4
  // CHECK-NEXT:    [[IDX2:%.*]] = add i64 [[IDX1]], [[I_ADD_EXT]]
  // NOOPT-NEXT:    [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8{{$}}
  // NOOPT-NEXT:    [[MAT:%.*]] = load <16 x double>, ptr [[MAT_ADDR]], align 8{{$}}
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 16
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // OPT-NEXT:      [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[MAT:%.*]] = load <16 x double>, ptr [[MAT_ADDR]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]]  = extractelement <16 x double> [[MAT]], i64 [[IDX2]]
  // CHECK-NEXT:    ret double [[MATEXT]]
  return m[i + 1][j - 1];
}

template <class T, unsigned R, unsigned C>
using matrix_type = T __attribute__((matrix_type(R, C)));
struct identmatrix_t {
  template <class T, unsigned N>
  operator matrix_type<T, N, N>() const {
    matrix_type<T, N, N> result;
    for (unsigned i = 0; i != N; ++i)
      result[i][i] = 1;
    return result;
  }
};

constexpr identmatrix_t identmatrix;

void test_constexpr1(matrix_type<float, 4, 4> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z15test_constexpr1Ru11matrix_typeILm4ELm4EfE(
  // NOOPT:         [[MAT:%.*]] = load <16 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <16 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[IM:%.*]] = call noundef <16 x float> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIfLj4EEEv(ptr {{[^,]*}} @_ZL11identmatrix)
  // CHECK-NEXT:    [[ADD:%.*]] = fadd <16 x float> [[MAT]], [[IM]]
  // NOOPT-NEXT:    [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8{{$}}
  // OPT-NEXT:      [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    store <16 x float> [[ADD]], ptr [[MAT_ADDR]], align 4
  // CHECK-NEXT:    ret voi

  // CHECK-LABEL: define linkonce_odr noundef <16 x float> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIfLj4EEEv(
  // CHECK-LABEL: for.body:                                         ; preds = %for.cond
  // NOOPT-NEXT:   [[I:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT-NEXT:     [[I:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[I_EXT:%.*]] = zext i32 [[I]] to i64
  // NOOPT-NEXT:   [[I2:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT-NEXT:     [[I2:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[I2_EXT:%.*]] = zext i32 [[I2]] to i64
  // CHECK-NEXT:   [[IDX1:%.*]] = mul i64 [[I2_EXT]], 4
  // CHECK-NEXT:   [[IDX2:%.*]] = add i64 [[IDX1]], [[I_EXT]]
  // OPT-NEXT:     [[CMP:%.*]] = icmp ult i64 [[IDX2]], 16
  // OPT-NEXT:     call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:   [[MAT:%.*]] = load <16 x float>, ptr %result, align 4{{$}}
  // CHECK-NEXT:   [[MATINS:%.*]] = insertelement <16 x float> [[MAT]], float 1.000000e+00, i64 [[IDX2]]
  // CHECK-NEXT:   store <16 x float> [[MATINS]], ptr %result, align 4
  // CHECK-NEXT:   br label %for.inc
  m = m + identmatrix;
}

void test_constexpr2(matrix_type<int, 5, 5> &m) {
  // CHECK-LABEL: define{{.*}} void @_Z15test_constexpr2Ru11matrix_typeILm5ELm5EiE(
  // CHECK:         [[IM:%.*]] = call noundef <25 x i32> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIiLj5EEEv(ptr {{[^,]*}} @_ZL11identmatrix)
  // NOOPT:         [[MAT:%.*]] = load <25 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <25 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[SUB:%.*]] = sub <25 x i32> [[IM]], [[MAT]]
  // CHECK-NEXT:    [[SUB2:%.*]] = add <25 x i32> [[SUB]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // NOOPT-NEXT:    [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8{{$}}
  // OPT-NEXT:      [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    store <25 x i32> [[SUB2]], ptr [[MAT_ADDR]], align 4
  // CHECK-NEXT:    ret void
  //

  // CHECK-LABEL: define linkonce_odr noundef <25 x i32> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIiLj5EEEv(
  // CHECK-LABEL: for.body:                                         ; preds = %for.cond
  // NOOPT-NEXT:   [[I:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT-NEXT:     [[I:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[I_EXT:%.*]] = zext i32 [[I]] to i64
  // NOOPT-NEXT:   [[I2:%.*]] = load i32, ptr %i, align 4{{$}}
  // OPT-NEXT:     [[I2:%.*]] = load i32, ptr %i, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:   [[I2_EXT:%.*]] = zext i32 [[I2]] to i64
  // CHECK-NEXT:   [[IDX1:%.*]] = mul i64 [[I2_EXT]], 5
  // CHECK-NEXT:   [[IDX2:%.*]] = add i64 [[IDX1]], [[I_EXT]]
  // OPT-NEXT:     [[CMP:%.*]] = icmp ult i64 [[IDX2]], 25
  // OPT-NEXT:     call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:   [[MAT:%.*]] = load <25 x i32>, ptr %result, align 4{{$}}
  // CHECK-NEXT:   [[MATINS:%.*]] = insertelement <25 x i32> [[MAT]], i32 1, i64 [[IDX2]]
  // CHECK-NEXT:   store <25 x i32> [[MATINS]], ptr %result, align 4
  // CHECK-NEXT:   br label %for.inc

  m = identmatrix - m + 1;
}
