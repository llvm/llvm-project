// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Also check we do not crash when running some middle-end passes. Most
// importantly this includes the IR verifier, to ensure we emit valid IR.
// RUN: %clang_cc1 -fenable-matrix -emit-llvm -triple x86_64-apple-darwin %s -o %t

// Tests for the matrix type builtins.

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
typedef float fx2x3_t __attribute__((matrix_type(2, 3)));
typedef float fx3x2_t __attribute__((matrix_type(3, 2)));
typedef int ix20x4_t __attribute__((matrix_type(20, 4)));
typedef int ix4x20_t __attribute__((matrix_type(4, 20)));
typedef unsigned ux1x6_t __attribute__((matrix_type(1, 6)));
typedef unsigned ux6x1_t __attribute__((matrix_type(6, 1)));

void transpose_double_5x5(dx5x5_t *a) {
  // CHECK-LABEL: define{{.*}} void @transpose_double_5x5(
  // CHECK:        [[A:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:   [[TRANS:%.*]] = call <25 x double> @llvm.matrix.transpose.v25f64(<25 x double> [[A]], i32 5, i32 5)
  // CHECK-NEXT:   [[AT_ADDR:%.*]] = bitcast [25 x double]* %a_t to <25 x double>*
  // CHECK-NEXT:   store <25 x double> [[TRANS]], <25 x double>* [[AT_ADDR]], align 8
  dx5x5_t a_t = __builtin_matrix_transpose(*a);
}

void transpose_float_3x2(fx3x2_t *a) {
  // CHECK-LABEL: define{{.*}} void @transpose_float_3x2(
  // CHECK:        [[A:%.*]] = load <6 x float>, <6 x float>* {{.*}}, align 4
  // CHECK-NEXT:   [[TRANS:%.*]] = call <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[A]], i32 3, i32 2)
  // CHECK-NEXT:   [[AT_ADDR:%.*]] = bitcast [6 x float]* %a_t to <6 x float>*
  // CHECK-NEXT:   store <6 x float> [[TRANS]], <6 x float>* [[AT_ADDR]], align 4

  fx2x3_t a_t = __builtin_matrix_transpose(*a);
}

void transpose_int_20x4(ix20x4_t *a) {
  // CHECK-LABEL: define{{.*}} void @transpose_int_20x4(
  // CHECK:         [[A:%.*]] = load <80 x i32>, <80 x i32>* {{.*}}, align 4
  // CHECK-NEXT:    [[TRANS:%.*]] = call <80 x i32> @llvm.matrix.transpose.v80i32(<80 x i32> [[A]], i32 20, i32 4)
  // CHECK-NEXT:    [[AT_ADDR:%.*]] = bitcast [80 x i32]* %a_t to <80 x i32>*
  // CHECK-NEXT:    store <80 x i32> [[TRANS]], <80 x i32>* [[AT_ADDR]], align 4

  ix4x20_t a_t = __builtin_matrix_transpose(*a);
}

struct Foo {
  ux1x6_t in;
  ux6x1_t out;
};

void transpose_struct_member(struct Foo *F) {
  // CHECK-LABEL: define{{.*}} void @transpose_struct_member(
  // CHECK:         [[M:%.*]] = load <6 x i32>, <6 x i32>* {{.*}}, align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <6 x i32> @llvm.matrix.transpose.v6i32(<6 x i32> [[M]], i32 1, i32 6)
  // CHECK-NEXT:    [[F_ADDR:%.*]] = load %struct.Foo*, %struct.Foo** %F.addr, align 8
  // CHECK-NEXT:    [[OUT_PTR:%.*]] = getelementptr inbounds %struct.Foo, %struct.Foo* [[F_ADDR]], i32 0, i32 1
  // CHECK-NEXT:    [[OUT_PTR_C:%.*]] = bitcast [6 x i32]* [[OUT_PTR]] to <6 x i32>*
  // CHECK-NEXT:    store <6 x i32> [[M_T]], <6 x i32>* [[OUT_PTR_C]], align 4

  F->out = __builtin_matrix_transpose(F->in);
}

void transpose_transpose_struct_member(struct Foo *F) {
  // CHECK-LABEL: define{{.*}} void @transpose_transpose_struct_member(
  // CHECK:         [[M:%.*]] = load <6 x i32>, <6 x i32>* {{.*}}, align 4
  // CHECK-NEXT:    [[M_T:%.*]] = call <6 x i32> @llvm.matrix.transpose.v6i32(<6 x i32> [[M]], i32 1, i32 6)
  // CHECK-NEXT:    [[M_T2:%.*]] = call <6 x i32> @llvm.matrix.transpose.v6i32(<6 x i32> [[M_T]], i32 6, i32 1)
  // CHECK-NEXT:    [[F_ADDR:%.*]] = load %struct.Foo*, %struct.Foo** %F.addr, align 8
  // CHECK-NEXT:    [[IN_PTR:%.*]] = getelementptr inbounds %struct.Foo, %struct.Foo* [[F_ADDR]], i32 0, i32 0
  // CHECK-NEXT:    [[IN_PTR_C:%.*]] = bitcast [6 x i32]* [[IN_PTR]] to <6 x i32>*
  // CHECK-NEXT:    store <6 x i32> [[M_T2]], <6 x i32>* [[IN_PTR_C]], align 4

  F->in = __builtin_matrix_transpose(__builtin_matrix_transpose(F->in));
}

dx5x5_t get_matrix();

void transpose_rvalue() {
  // CHECK-LABEL: define{{.*}} void @transpose_rvalue()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[M_T_ADDR:%.*]] = alloca [25 x double], align 8
  // CHECK-NEXT:    [[CALL:%.*]] = call <25 x double> (...) @get_matrix()
  // CHECK-NEXT:    [[M_T:%.*]] = call <25 x double> @llvm.matrix.transpose.v25f64(<25 x double> [[CALL]], i32 5, i32 5)
  // CHECK-NEXT:    [[M_T_ADDR_C:%.*]] = bitcast [25 x double]* [[M_T_ADDR]] to <25 x double>*
  // CHECK-NEXT:    store <25 x double> [[M_T]], <25 x double>* [[M_T_ADDR_C]], align 8

  dx5x5_t m_t = __builtin_matrix_transpose(get_matrix());
}

const dx5x5_t global_matrix;

void transpose_global() {
  // CHECK-LABEL: define{{.*}} void @transpose_global()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[M_T_ADDR:%.*]] = alloca [25 x double], align 8
  // CHECK-NEXT:    [[GLOBAL_MATRIX:%.*]] = load <25 x double>, <25 x double>* bitcast ([25 x double]* @global_matrix to <25 x double>*), align 8
  // CHECK-NEXT:    [[M_T:%.*]] = call <25 x double> @llvm.matrix.transpose.v25f64(<25 x double> [[GLOBAL_MATRIX]], i32 5, i32 5)
  // CHECK-NEXT:    [[M_T_ADDR_C:%.*]] = bitcast [25 x double]* [[M_T_ADDR]] to <25 x double>*
  // CHECK-NEXT:    store <25 x double> [[M_T]], <25 x double>* [[M_T_ADDR_C]], align 8

  dx5x5_t m_t = __builtin_matrix_transpose(global_matrix);
}

void column_major_load_with_const_stride_double(double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @column_major_load_with_const_stride_double(double* %Ptr)
  // CHECK:         [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call <25 x double> @llvm.matrix.column.major.load.v25f64(double* align 8 [[PTR]], i64 5, i1 false, i32 5, i32 5)

  dx5x5_t m_a1 = __builtin_matrix_column_major_load(Ptr, 5, 5, 5);
}

void column_major_load_with_const_stride2_double(double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @column_major_load_with_const_stride2_double(double* %Ptr)
  // CHECK:         [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call <25 x double> @llvm.matrix.column.major.load.v25f64(double* align 8 [[PTR]], i64 15, i1 false, i32 5, i32 5)

  dx5x5_t m_a2 = __builtin_matrix_column_major_load(Ptr, 5, 5, 2 * 3 + 9);
}

void column_major_load_with_variable_stride_ull_float(float *Ptr, unsigned long long S) {
  // CHECK-LABEL: define{{.*}} void @column_major_load_with_variable_stride_ull_float(float* %Ptr, i64 %S)
  // CHECK:         [[S:%.*]] = load i64, i64* %S.addr, align 8
  // CHECK-NEXT:    [[PTR:%.*]] = load float*, float** %Ptr.addr, align 8
  // CHECK-NEXT:    call <6 x float> @llvm.matrix.column.major.load.v6f32(float* align 4 [[PTR]], i64 [[S]], i1 false, i32 2, i32 3)

  fx2x3_t m_b = __builtin_matrix_column_major_load(Ptr, 2, 3, S);
}

void column_major_load_with_stride_math_int(int *Ptr, int S) {
  // CHECK-LABEL: define{{.*}} void @column_major_load_with_stride_math_int(i32* %Ptr, i32 %S)
  // CHECK:         [[S:%.*]] = load i32, i32* %S.addr, align 4
  // CHECK-NEXT:    [[STRIDE:%.*]] = add nsw i32 [[S]], 32
  // CHECK-NEXT:    [[STRIDE_EXT:%.*]] = sext i32 [[STRIDE]] to i64
  // CHECK-NEXT:    [[PTR:%.*]] = load i32*, i32** %Ptr.addr, align 8
  // CHECK-NEXT:    call <80 x i32> @llvm.matrix.column.major.load.v80i32(i32* align 4 [[PTR]], i64 [[STRIDE_EXT]], i1 false, i32 4, i32 20)

  ix4x20_t m_c = __builtin_matrix_column_major_load(Ptr, 4, 20, S + 32);
}

void column_major_load_with_stride_math_s_int(int *Ptr, short S) {
  // CHECK-LABEL:  define{{.*}} void @column_major_load_with_stride_math_s_int(i32* %Ptr, i16 signext %S)
  // CHECK:         [[S:%.*]] = load i16, i16* %S.addr, align 2
  // CHECK-NEXT:    [[S_EXT:%.*]] = sext i16 [[S]] to i32
  // CHECK-NEXT:    [[STRIDE:%.*]] = add nsw i32 [[S_EXT]], 32
  // CHECK-NEXT:    [[STRIDE_EXT:%.*]] = sext i32 [[STRIDE]] to i64
  // CHECK-NEXT:    [[PTR:%.*]] = load i32*, i32** %Ptr.addr, align 8
  // CHECK-NEXT:    %matrix = call <80 x i32> @llvm.matrix.column.major.load.v80i32(i32* align 4 [[PTR]], i64 [[STRIDE_EXT]], i1 false, i32 4, i32 20)

  ix4x20_t m_c = __builtin_matrix_column_major_load(Ptr, 4, 20, S + 32);
}

void column_major_load_array1(double Ptr[25]) {
  // CHECK-LABEL: define{{.*}} void @column_major_load_array1(double* %Ptr)
  // CHECK:         [[ADDR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call <25 x double> @llvm.matrix.column.major.load.v25f64(double* align 8 [[ADDR]], i64 5, i1 false, i32 5, i32 5)

  dx5x5_t m = __builtin_matrix_column_major_load(Ptr, 5, 5, 5);
}

void column_major_load_array2() {
  // CHECK-LABEL: define{{.*}} void @column_major_load_array2() #0 {
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[PTR:%.*]] = alloca [25 x double], align 16
  // CHECK:         [[ARRAY_DEC:%.*]] = getelementptr inbounds [25 x double], [25 x double]* [[PTR]], i64 0, i64 0
  // CHECK-NEXT:    call <25 x double> @llvm.matrix.column.major.load.v25f64(double* align 16 [[ARRAY_DEC]], i64 5, i1 false, i32 5, i32 5)

  double Ptr[25];
  dx5x5_t m = __builtin_matrix_column_major_load(Ptr, 5, 5, 5);
}

void column_major_load_const(const double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @column_major_load_const(double* %Ptr)
  // CHECK:         [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call <25 x double> @llvm.matrix.column.major.load.v25f64(double* align 8 [[PTR]], i64 5, i1 false, i32 5, i32 5)

  dx5x5_t m_a1 = __builtin_matrix_column_major_load(Ptr, 5, 5, 5);
}

void column_major_load_volatile(volatile double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @column_major_load_volatile(double* %Ptr)
  // CHECK:         [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call <25 x double> @llvm.matrix.column.major.load.v25f64(double* align 8 [[PTR]], i64 5, i1 true, i32 5, i32 5)

  dx5x5_t m_a1 = __builtin_matrix_column_major_load(Ptr, 5, 5, 5);
}

void column_major_store_with_const_stride_double(double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @column_major_store_with_const_stride_double(double* %Ptr)
  // CHECK:         [[M:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:    [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v25f64(<25 x double> [[M]], double* align 8 [[PTR]], i64 5, i1 false, i32 5, i32 5)

  dx5x5_t m;
  __builtin_matrix_column_major_store(m, Ptr, 5);
}

void column_major_store_with_const_stride2_double(double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @column_major_store_with_const_stride2_double(double* %Ptr)
  // CHECK:         [[M:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:    [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v25f64(<25 x double> [[M]], double* align 8 [[PTR]], i64 15, i1 false, i32 5, i32 5)
  //
  dx5x5_t m;
  __builtin_matrix_column_major_store(m, Ptr, 2 * 3 + 9);
}

void column_major_store_with_stride_math_int(int *Ptr, int S) {
  // CHECK-LABEL: define{{.*}} void @column_major_store_with_stride_math_int(i32* %Ptr, i32 %S)
  // CHECK:         [[M:%.*]] = load <80 x i32>, <80 x i32>* {{.*}}, align 4
  // CHECK-NEXT:    [[PTR:%.*]] = load i32*, i32** %Ptr.addr, align 8
  // CHECK-NEXT:    [[S:%.*]] = load i32, i32* %S.addr, align 4
  // CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[S]], 32
  // CHECK-NEXT:    [[IDX:%.*]] = sext i32 [[ADD]] to i64
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v80i32(<80 x i32> [[M]], i32* align 4 [[PTR]], i64 [[IDX]], i1 false, i32 4, i32 20)

  ix4x20_t m;
  __builtin_matrix_column_major_store(m, Ptr, S + 32);
}

void column_major_store_with_stride_math_s_int(int *Ptr, short S) {
  // CHECK-LABEL: define{{.*}} void @column_major_store_with_stride_math_s_int(i32* %Ptr, i16 signext %S)
  // CHECK:         [[M:%.*]] = load <80 x i32>, <80 x i32>* {{.*}}, align 4
  // CHECK-NEXT:    [[PTR:%.*]] = load i32*, i32** %Ptr.addr, align 8
  // CHECK-NEXT:    [[S:%.*]] = load i16, i16* %S.addr, align 2
  // CHECK-NEXT:    [[EXT:%.*]] = sext i16 [[S]] to i32
  // CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[EXT]], 2
  // CHECK-NEXT:    [[IDX:%.*]] = sext i32 [[ADD]] to i64
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v80i32(<80 x i32> [[M]], i32* align 4 [[PTR]], i64 [[IDX]], i1 false, i32 4, i32 20)

  ix4x20_t m;
  __builtin_matrix_column_major_store(m, Ptr, S + 2);
}

void column_major_store_array1(double Ptr[25]) {
  // CHECK-LABEL: define{{.*}} void @column_major_store_array1(double* %Ptr)
  // CHECK:         [[M:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:    [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v25f64(<25 x double> [[M]], double* align 8 [[PTR]], i64 5, i1 false, i32 5, i32 5)

  dx5x5_t m;
  __builtin_matrix_column_major_store(m, Ptr, 5);
}

void column_major_store_array2() {
  // CHECK-LABEL: define{{.*}} void @column_major_store_array2()
  // CHECK:         [[M:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:    [[PTR:%.*]] = getelementptr inbounds [25 x double], [25 x double]* %Ptr, i64 0, i64 0
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v25f64(<25 x double> [[M]], double* align 16 [[PTR]], i64 5, i1 false, i32 5, i32 5)

  double Ptr[25];
  dx5x5_t m;
  __builtin_matrix_column_major_store(m, Ptr, 5);
}

void column_major_store_volatile(volatile double *Ptr) {
  // CHECK-LABEL: define{{.*}} void @column_major_store_volatile(double* %Ptr) #0 {
  // CHECK:         [[M:%.*]] = load <25 x double>, <25 x double>* {{.*}}, align 8
  // CHECK-NEXT:    [[PTR:%.*]] = load double*, double** %Ptr.addr, align 8
  // CHECK-NEXT:    call void @llvm.matrix.column.major.store.v25f64(<25 x double> [[M]], double* align 8 [[PTR]], i64 5, i1 true, i32 5, i32 5)

  dx5x5_t m;
  __builtin_matrix_column_major_store(m, Ptr, 5);
}
