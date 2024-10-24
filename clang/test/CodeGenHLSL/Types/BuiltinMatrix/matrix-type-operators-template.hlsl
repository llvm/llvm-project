// RUN: %clang_cc1 -O0 -triple spirv-unknown-vulkan-compute -std=hlsl202y -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s --check-prefixes=CHECK,NOOPT
// RUN: %clang_cc1 -O1 -triple spirv-unknown-vulkan-compute -std=hlsl202y -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s --check-prefixes=CHECK,OPT

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  using matrix_t = matrix<EltTy, Rows, Columns>;

  matrix_t value;
};

template <typename EltTy0, unsigned R0, unsigned C0>
typename MyMatrix<EltTy0, R0, C0>::matrix_t add(inout MyMatrix<EltTy0, R0, C0> A, inout MyMatrix<EltTy0, R0, C0> B) {
  return A.value + B.value;
}

// CHECK-LABEL: define {{.*}}test_add_template
void test_add_template() {
  // CHECK:       call{{.*}} <8 x float> @_Z3addIfLj2ELj4EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tES2_S2_(ptr noalias noundef nonnull align 4 dereferenceable(32) %{{.*}}, ptr noalias noundef nonnull align 4 dereferenceable(32) %{{.*}})

  // CHECK-LABEL: define{{.*}} <8 x float> @_Z3addIfLj2ELj4EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tES2_S2_(
  // NOOPT:       [[MAT1:%.*]] = load <8 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT:       [[MAT2:%.*]] = load <8 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[MAT1:%.*]] = load <8 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:         [[MAT2:%.*]] = load <8 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fadd <8 x float> [[MAT1]], [[MAT2]]
  // CHECK-NEXT:  ret <8 x float> [[RES]]

  MyMatrix<float, 2, 4> Mat1;
  MyMatrix<float, 2, 4> Mat2;
  Mat1.value = add(Mat1, Mat2);
}

template <typename EltTy0, unsigned R0, unsigned C0>
typename MyMatrix<EltTy0, R0, C0>::matrix_t subtract(inout MyMatrix<EltTy0, R0, C0> A, inout MyMatrix<EltTy0, R0, C0> B) {
  return A.value - B.value;
}

// CHECK-LABEL: define {{.*}}test_subtract_template
void test_subtract_template() {
  // CHECK:       call{{.*}} <8 x float> @_Z8subtractIfLj2ELj4EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tES2_S2_(ptr noalias noundef nonnull align 4 dereferenceable(32) %{{.*}}, ptr noalias noundef nonnull align 4 dereferenceable(32) %{{.*}})

  // CHECK-LABEL: define{{.*}} <8 x float> @_Z8subtractIfLj2ELj4EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tES2_S2_(
  // NOOPT:       [[MAT1:%.*]] = load <8 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT:       [[MAT2:%.*]] = load <8 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:         [[MAT1:%.*]] = load <8 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:         [[MAT2:%.*]] = load <8 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[RES:%.*]] = fsub <8 x float> [[MAT1]], [[MAT2]]
  // CHECK-NEXT:  ret <8 x float> [[RES]]

  MyMatrix<float, 2, 4> Mat1;
  MyMatrix<float, 2, 4> Mat2;
  Mat1.value = subtract(Mat1, Mat2);
}

struct DoubleWrapper1 {
  int x;
  operator double() {
    return x;
  }
};

// CHECK-LABEL: define {{.*}}test_DoubleWrapper1_Sub1
void test_DoubleWrapper1_Sub1(inout MyMatrix<double, 4, 3> m) {
  // NOOPT:       [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR:%.*]] = call{{.*}} double @_ZN14DoubleWrapper1cvdEv(ptr {{[^,]*}} %w1)
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x double> [[SCALAR_EMBED]], <12 x double> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <12 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK:       store <12 x double> [[RES]], ptr {{.*}}, align 8

  DoubleWrapper1 w1;
  w1.x = 10;
  m.value = m.value - w1;
}

// CHECK-LABEL: define {{.*}}test_DoubleWrapper1_Sub2
void test_DoubleWrapper1_Sub2(inout MyMatrix<double, 4, 3> m) {
  // CHECK:       [[SCALAR:%.*]] = call{{.*}} double @_ZN14DoubleWrapper1cvdEv(ptr {{[^,]*}} %w1)
  // NOOPT:       [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x double> [[SCALAR_EMBED]], <12 x double> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <12 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <12 x double> [[RES]], ptr {{.*}}, align 8

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

// CHECK-LABEL: define {{.*}}test_DoubleWrapper2_Add1
void test_DoubleWrapper2_Add1(inout MyMatrix<double, 4, 3> m) {
  // NOOPT:       [[MATRIX:%.*]] = load <12 x double>, ptr {{.+}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <12 x double>, ptr {{.+}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK:       [[SCALAR:%.*]] = call{{.*}} double @_ZN14DoubleWrapper2cvdEv(ptr {{[^,]*}} %w2)
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x double> [[SCALAR_EMBED]], <12 x double> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <12 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK:       store <12 x double> [[RES]], ptr {{.*}}, align 8

  DoubleWrapper2 w2;
  w2.x = 20;
  m.value = m.value + w2;
}

// CHECK-LABEL: define {{.*}}test_DoubleWrapper2_Add2
void test_DoubleWrapper2_Add2(inout MyMatrix<double, 4, 3> m) {
  // CHECK:       [[SCALAR:%.*]] = call{{.*}} double @_ZN14DoubleWrapper2cvdEv(ptr {{[^,]*}} %w2)
  // NOOPT:       [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x double> poison, double [[SCALAR]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x double> [[SCALAR_EMBED]], <12 x double> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <12 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <12 x double> [[RES]], ptr {{.*}}, align 8

  DoubleWrapper2 w2;
  w2.x = 20;
  m.value = w2 + m.value;
}

struct IntWrapper {
  uint16_t x;
  operator int() {
    return x;
  }
};

// CHECK-LABEL: define {{.*}}test_IntWrapper_Add
void test_IntWrapper_Add(inout MyMatrix<double, 4, 3> m) {
  // NOOPT:       [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR:%.*]] = call{{.*}} i32 @_ZN10IntWrappercviEv(ptr {{[^,]*}} %w3)
  // CHECK-NEXT:  [[SCALAR_FP:%.*]] = sitofp i32 [[SCALAR]] to double
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x double> poison, double [[SCALAR_FP]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x double> [[SCALAR_EMBED]], <12 x double> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fadd <12 x double> [[MATRIX]], [[SCALAR_EMBED1]]
  // CHECK:       store <12 x double> [[RES]], ptr {{.*}}, align 8

  IntWrapper w3;
  w3.x = 13;
  m.value = m.value + w3;
}

// CHECK-LABEL: define {{.*}}test_IntWrapper_Sub
void test_IntWrapper_Sub(inout MyMatrix<double, 4, 3> m) {
  // CHECK:       [[SCALAR:%.*]] = call{{.*}} i32 @_ZN10IntWrappercviEv(ptr {{[^,]*}} %w3)
  // CHECK-NEXT:  [[SCALAR_FP:%.*]] = sitofp i32 [[SCALAR]] to double
  // NOOPT:       [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x double> poison, double [[SCALAR_FP]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x double> [[SCALAR_EMBED]], <12 x double> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fsub <12 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <12 x double> [[RES]], ptr {{.*}}, align 8

  IntWrapper w3;
  w3.x = 13;
  m.value = w3 - m.value;
}

template <typename EltTy0, unsigned R0, unsigned C0, unsigned C1>
typename MyMatrix<EltTy0, R0, C1>::matrix_t multiply(inout MyMatrix<EltTy0, R0, C0> A, inout MyMatrix<EltTy0, C0, C1> B) {
  return A.value * B.value;
}

// CHECK-LABEL: define {{.*}}test_multiply_template
MyMatrix<float, 2, 2> test_multiply_template(MyMatrix<float, 2, 4> Mat1,
                                             MyMatrix<float, 4, 2> Mat2) {
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %tmp = alloca %struct.MyMatrix, align 4
  // CHECK-NEXT:    %tmp1 = alloca %struct.MyMatrix.2, align 4
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp, ptr align 4 %Mat1, i64 32, i1 false)
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 32, ptr %tmp)
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp1, ptr align 4 %Mat2, i64 32, i1 false)
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 32, ptr %tmp1)
  // CHECK-NEXT:    [[RES:%.*]] = call{{.*}} <4 x float> @_Z8multiplyIfLj2ELj4ELj2EEN8MyMatrixIT_XT0_EXT2_EE8matrix_tES0_IS1_XT0_EXT1_EES0_IS1_XT1_EXT2_EE(ptr noalias noundef nonnull align 4 dereferenceable(32) %tmp, ptr noalias noundef nonnull align 4 dereferenceable(32) %tmp1)
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %Mat1, ptr align 4 %tmp, i64 32, i1 false)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 32, ptr %tmp)
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %Mat2, ptr align 4 %tmp1, i64 32, i1 false)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 32, ptr %tmp1)
  // CHECK-NEXT:    %value = getelementptr inbounds nuw %struct.MyMatrix.1, ptr %agg.result, i32 0, i32 0
  // CHECK-NEXT:    store <4 x float> [[RES]], ptr %value, align 4
  // CHECK-NEXT:    ret void
  //
  // CHECK-LABEL:  define{{.*}} <4 x float> @_Z8multiplyIfLj2ELj4ELj2EEN8MyMatrixIT_XT0_EXT2_EE8matrix_tES0_IS1_XT0_EXT1_EES0_IS1_XT1_EXT2_EE(
  // NOOPT:         [[MAT1:%.*]] = load <8 x float>, ptr {{.*}}, align 4{{$}}
  // NOOPT:         [[MAT2:%.*]] = load <8 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT1:%.*]] = load <8 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT:           [[MAT2:%.*]] = load <8 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[RES:%.*]] = call <4 x float> @llvm.matrix.multiply.v4f32.v8f32.v8f32(<8 x float> [[MAT1]], <8 x float> [[MAT2]], i32 2, i32 4, i32 2)
  // CHECK-NEXT:    ret <4 x float> [[RES]]

  MyMatrix<float, 2, 2> Res;
  Res.value = multiply(Mat1, Mat2);
  return Res;
}

// CHECK-LABEL: define {{.*}}test_IntWrapper_Multiply
void test_IntWrapper_Multiply(inout MyMatrix<double, 4, 3> m, inout IntWrapper w3) {
  // CHECK:       [[SCALAR:%.*]] = call{{.*}} i32 @_ZN10IntWrappercviEv(ptr noundef {{.*}})
  // CHECK-NEXT:  [[SCALAR_FP:%.*]] = sitofp i32 [[SCALAR]] to double
  // NOOPT:       [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:         [[MATRIX:%.*]] = load <12 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:  [[SCALAR_EMBED:%.*]] = insertelement <12 x double> poison, double [[SCALAR_FP]], i64 0
  // CHECK-NEXT:  [[SCALAR_EMBED1:%.*]] = shufflevector <12 x double> [[SCALAR_EMBED]], <12 x double> poison, <12 x i32> zeroinitializer
  // CHECK-NEXT:  [[RES:%.*]] = fmul <12 x double> [[SCALAR_EMBED1]], [[MATRIX]]
  // CHECK:       store <12 x double> [[RES]], ptr {{.*}}, align 8
  // CHECK-NEXT:  ret void
  m.value = w3 * m.value;
}

template <typename EltTy, unsigned Rows, unsigned Columns>
void insert(inout MyMatrix<EltTy, Rows, Columns> Mat, EltTy e, unsigned i, unsigned j) {
  Mat.value[i][j] = e;
}

// CHECK-LABEL: define {{.*}}test_insert_template1
void test_insert_template1(inout MyMatrix<unsigned, 2, 2> Mat, unsigned e, unsigned i, unsigned j) {
  // NOOPT:         [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8{{$}}
  // NOOPT:         [[E:%.*]] = load i32, ptr %e.addr, align 4{{$}}
  // NOOPT-NEXT:    [[I:%.*]] = load i32, ptr %i.addr, align 4{{$}}
  // NOOPT-NEXT:    [[J:%.*]] = load i32, ptr %j.addr, align 4{{$}}
  // OPT:           [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT:           [[E:%.*]] = load i32, ptr %e.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[I:%.*]] = load i32, ptr %i.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // OPT-NEXT:      [[J:%.*]] = load i32, ptr %j.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    call{{.*}} void @_Z6insertIjLj2ELj2EEv8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noalias noundef nonnull align 4 dereferenceable(16) %{{.*}}, i32 noundef [[E]], i32 noundef [[I]], i32 noundef [[J]])
  // CHECK:         ret void
  //
  // CHECK-LABEL: define{{.*}} void @_Z6insertIjLj2ELj2EEv8MyMatrixIT_XT0_EXT1_EES1_jj(
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

// CHECK-LABEL: define {{.*}}test_insert_template2
void test_insert_template2(inout MyMatrix<float, 3, 4> Mat, float e) {
  // NOOPT:         [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8{{$}}
  // NOOPT:         [[E:%.*]] = load float, ptr %e.addr, align 4{{$}}
  // OPT:           [[MAT_ADDR:%.*]] = load ptr, ptr %Mat.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // OPT:           [[E:%.*]] = load float, ptr %e.addr, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    call{{.*}} void @_Z6insertIfLj3ELj4EEv8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noalias noundef nonnull align 4 dereferenceable(48) %{{.*}}, float noundef [[E]], i32 noundef 2, i32 noundef 3)
  // CHECK:         ret void
  //
  // CHECK-LABEL: define{{.*}} void @_Z6insertIfLj3ELj4EEv8MyMatrixIT_XT0_EXT1_EES1_jj(
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
  // OPT-NEXT:      [[CMP:%.*]] = icmp ult i64 [[IDX2]], 12
  // OPT-NEXT:      call void @llvm.assume(i1 [[CMP]])
  // CHECK-NEXT:    [[MAT:%.*]] = load <12 x float>, ptr {{.*}}, align 4{{$}}
  // CHECK-NEXT:    [[MATINS:%.*]] = insertelement <12 x float> [[MAT]], float [[E]], i64 [[IDX2]]
  // CHECK-NEXT:    store <12 x float> [[MATINS]], ptr {{.*}}, align 4
  // CHECK-NEXT:    ret void

  insert(Mat, e, 2, 3);
}

template <typename EltTy, unsigned Rows, unsigned Columns>
EltTy extract(inout MyMatrix<EltTy, Rows, Columns> Mat) {
  return Mat.value[1u][0u];
}

// CHECK-LABEL: define {{.*}}test_extract_template
int test_extract_template(MyMatrix<int, 2, 2> Mat1) {
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %tmp = alloca %struct.MyMatrix.5, align 4
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp, ptr align 4 %Mat1, i64 16, i1 false)
  // OPT-NEXT:      call void @llvm.lifetime.start.p0(i64 16, ptr %tmp)
  // CHECK-NEXT:    [[CALL:%.*]] = call{{.*}} i32 @_Z7extractIiLj2ELj2EET_8MyMatrixIS0_XT0_EXT1_EE(ptr noalias noundef nonnull align 4 dereferenceable(16) %tmp)
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %Mat1, ptr align 4 %tmp, i64 16, i1 false)
  // OPT-NEXT:      call void @llvm.lifetime.end.p0(i64 16, ptr %tmp)
  // CHECK-NEXT:    ret i32 [[CALL]]
  //
  // CHECK-LABEL: define{{.*}} i32 @_Z7extractIiLj2ELj2EET_8MyMatrixIS0_XT0_EXT1_EE(
  // NOOPT:         [[MAT:%.*]] = load <4 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <4 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <4 x i32> [[MAT]], i64 1
  // CHECK-NEXT:    ret i32 [[MATEXT]]

  return extract(Mat1);
}

template <class R, class C>
auto matrix_subscript(double4x4 m, R r, C c) -> decltype(m[r][c]) {}

// CHECK-LABEL: define {{.*}}test_matrix_subscript
double test_matrix_subscript(double4x4 m) {
  // NOOPT:         [[MAT:%.*]] = load <16 x double>, ptr {{.*}}, align 8{{$}}
  // OPT:           [[MAT:%.*]] = load <16 x double>, ptr {{.*}}, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[CALL:%.*]] = call{{.*}} nonnull align 8 dereferenceable(8) ptr @_Z16matrix_subscriptIiiEDTixixfp_fp0_fp1_Eu11matrix_typeILj4ELj4EdET_T0_(<16 x double> noundef [[MAT]], i32 noundef 1, i32 noundef 2)
  // NOOPT-NEXT:    [[RES:%.*]] = load double, ptr [[CALL]], align 8{{$}}
  // OPT-NEXT:      [[RES:%.*]] = load double, ptr [[CALL]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    ret double [[RES]]

  return matrix_subscript(m, 1, 2);
}

// CHECK-LABEL: define {{.*}}test_matrix_subscript_const
const double test_matrix_subscript_const(const double4x4 m) {
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    [[M_ADDR:%.*]] = alloca [16 x double], align 8
  // CHECK-NEXT:    store <16 x double> [[M:%.*]], ptr [[M_ADDR]], align 8
  // NOOPT:         [[NAMELESS1:%.*]] = load <16 x double>, ptr [[M_ADDR]], align 8{{$}}
  // OPT:           [[NAMELESS1:%.*]] = load <16 x double>, ptr [[M_ADDR]], align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[MATEXT:%.*]] = extractelement <16 x double> [[NAMELESS1]], i64 4
  // CHECK-NEXT:    ret double [[MATEXT]]

  return m[0][1];
}

struct UnsignedWrapper {
  char x;
  operator unsigned() {
    return x;
  }
};

// CHECK-LABEL: define {{.*}}extract_IntWrapper_idx
double extract_IntWrapper_idx(inout double4x4 m, IntWrapper i, UnsignedWrapper j) {
  // CHECK:         [[I:%.*]] = call{{.*}} i32 @_ZN10IntWrappercviEv(ptr {{[^,]*}} %i)
  // CHECK-NEXT:    [[I_ADD:%.*]] = add nsw i32 [[I]], 1
  // CHECK-NEXT:    [[I_ADD_EXT:%.*]] = sext i32 [[I_ADD]] to i64
  // CHECK-NEXT:    [[J:%.*]] = call{{.*}} i32 @_ZN15UnsignedWrappercvjEv(ptr {{[^,]*}} %j)
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
using matrix_type = matrix<T, R, C>;
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

// CHECK-LABEL: define {{.*}}test_constexpr1
void test_constexpr1(inout matrix_type<float, 4, 4> m) {
  // NOOPT:         [[MAT:%.*]] = load <16 x float>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <16 x float>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[IM:%.*]] = call{{.*}} <16 x float> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIfLj4EEEv(ptr {{[^,]*}} @_ZL11identmatrix)
  // CHECK-NEXT:    [[ADD:%.*]] = fadd <16 x float> [[MAT]], [[IM]]
  // NOOPT-NEXT:    [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8{{$}}
  // OPT-NEXT:      [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    store <16 x float> [[ADD]], ptr [[MAT_ADDR]], align 4
  // CHECK-NEXT:    ret voi

  // CHECK-LABEL: define{{.*}} <16 x float> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIfLj4EEEv(
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

// CHECK-LABEL: define {{.*}}test_constexpr2
void test_constexpr2(inout matrix_type<int, 4, 4> m) {
  // CHECK:         [[IM:%.*]] = call{{.*}} <16 x i32> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIiLj4EEEv(ptr {{[^,]*}} @_ZL11identmatrix)
  // NOOPT:         [[MAT:%.*]] = load <16 x i32>, ptr {{.*}}, align 4{{$}}
  // OPT:           [[MAT:%.*]] = load <16 x i32>, ptr {{.*}}, align 4, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    [[SUB:%.*]] = sub <16 x i32> [[IM]], [[MAT]]
  // CHECK-NEXT:    [[SUB2:%.*]] = add <16 x i32> [[SUB]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // NOOPT-NEXT:    [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8{{$}}
  // OPT-NEXT:      [[MAT_ADDR:%.*]] = load ptr, ptr %m.addr, align 8, !tbaa !{{[0-9]+}}{{$}}
  // CHECK-NEXT:    store <16 x i32> [[SUB2]], ptr [[MAT_ADDR]], align 4
  // CHECK-NEXT:    ret void
  //

  // CHECK-LABEL: define{{.*}} <16 x i32> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIiLj4EEEv(
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
  // CHECK-NEXT:   [[MAT:%.*]] = load <16 x i32>, ptr %result, align 4{{$}}
  // CHECK-NEXT:   [[MATINS:%.*]] = insertelement <16 x i32> [[MAT]], i32 1, i64 [[IDX2]]
  // CHECK-NEXT:   store <16 x i32> [[MATINS]], ptr %result, align 4
  // CHECK-NEXT:   br label %for.inc

  m = identmatrix - m + 1;
}
