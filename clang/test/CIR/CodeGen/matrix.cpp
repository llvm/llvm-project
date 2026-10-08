// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fenable-matrix -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fenable-matrix -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fenable-matrix -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-OGCG

typedef float matrix3x3 __attribute__((matrix_type(3, 3)));
typedef float matrix3x2 __attribute__((matrix_type(3, 2)));
typedef float matrix2x3 __attribute__((matrix_type(2, 3)));

matrix3x3 a;

// CIR: cir.global external @a = #cir.zero : !cir.matrix<3 x 3 x !cir.float>
// LLVM: @a = global [9 x float] zeroinitializer, align 4

void local_matrix() {
  matrix3x3 a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// LLVM: %[[A_ADDR:.*]] = alloca [9 x float], align 4

void load_and_store() {
  matrix3x3 a;
  matrix3x3 b;
  b = a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[TMP_A:.*]] = load <9 x float>, ptr %[[A_ADDR]], align 4
// LLVM: store <9 x float> %[[TMP_A]], ptr %[[B_ADDR]], align 4

void load_global_store_in_local() {
  matrix3x3 b;
  b = a;
}

// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[GLOBAL_A:.*]] = cir.get_global @a : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[GLOBAL_A]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

// LLVM: %[[B_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[TMP_A:.*]] = load <9 x float>, ptr @a, align 4
// LLVM: store <9 x float> %[[TMP_A]], ptr %[[B_ADDR]], align 4

void builtin_matrix_transpose() {
  matrix3x3 a;
  matrix3x3 b = __builtin_matrix_transpose(a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: %[[TRANSPOSE:.*]] = cir.matrix.transpose %[[TMP_A]] : <3 x 3 x !cir.float>, !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[TRANSPOSE]], %[[B_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[TMP_A:.*]] = load <9 x float>, ptr %[[A_ADDR]], align 4
// LLVM: %[[TRANSPOSE:.*]] = call <9 x float> @llvm.matrix.transpose.v9f32(<9 x float> %[[TMP_A]], i32 3, i32 3)
// LLVM: store <9 x float> %[[TRANSPOSE]], ptr %[[B_ADDR]], align 4

void builtin_matrix_transpose_different_sizes() {
  matrix3x2 a;
  matrix2x3 b = __builtin_matrix_transpose(a);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.matrix<2 x 3 x !cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.matrix<3 x 2 x !cir.float>>, !cir.matrix<3 x 2 x !cir.float>
// CIR: %[[TRANSPOSE:.*]] = cir.matrix.transpose %[[TMP_A]] : <3 x 2 x !cir.float>, !cir.matrix<2 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[TRANSPOSE]], %[[B_ADDR]] : !cir.matrix<2 x 3 x !cir.float>, !cir.ptr<!cir.matrix<2 x 3 x !cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca [6 x float], align 4
// LLVM: %[[B_ADDR:.*]] = alloca [6 x float], align 4
// LLVM: %[[TMP_A:.*]] = load <6 x float>, ptr %[[A_ADDR]], align 4
// LLVM: %[[TRANSPOSE:.*]] = call <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %[[TMP_A]], i32 3, i32 2)
// LLVM: store <6 x float> %[[TRANSPOSE:.*]], ptr %[[B_ADDR]], align 4

void column_major_load() {
  float *ptr;
  matrix3x3 matrix = __builtin_matrix_column_major_load(ptr, 3, 3, 3);
}

// CIR: %[[PTR_ADDR:.*]] = cir.alloca "ptr" {{.*}} : !cir.ptr<!cir.ptr<!cir.float>>
// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} init : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[STRIDE:.*]] = cir.const #cir.int<3> : !u64i
// CIR: %[[TMP_PTR:.*]] = cir.load {{.*}} %[[PTR_ADDR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CIR: %[[RESULT:.*]] = cir.matrix.column_major_load %[[TMP_PTR]] : <!cir.float>, %[[STRIDE]] : !u64i, !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[RESULT]], %[[MATRIX_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

// LLVM: %[[PTR_ADDR:.*]] = alloca ptr, align 8
// LLVM: %[[MATRIX_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[TMP_PTR:.*]] = load ptr, ptr %[[PTR_ADDR]], align 8
// LLVM: %[[RESULT:.*]] = call <9 x float> @llvm.matrix.column.major.load.v9f32.i64(ptr align 4 %[[TMP_PTR]], i64 3, i1 false, i32 3, i32 3)
// LLVM: store <9 x float> %[[RESULT]], ptr %[[MATRIX_ADDR]], align 4

void column_major_volatile_load() {
  volatile float *ptr;
  matrix3x3 matrix = __builtin_matrix_column_major_load(ptr, 3, 3, 3);
}

// CIR: %[[PTR_ADDR:.*]] = cir.alloca "ptr" {{.*}} : !cir.ptr<!cir.ptr<!cir.float>>
// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} init : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[STRIDE:.*]] = cir.const #cir.int<3> : !u64i
// CIR: %[[TMP_PTR:.*]] = cir.load {{.*}} %[[PTR_ADDR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CIR: %[[RESULT:.*]] = cir.matrix.column_major_load %[[TMP_PTR]] : <!cir.float>, %[[STRIDE]] : !u64i volatile, !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[RESULT]], %[[MATRIX_ADDR]] : !cir.matrix<3 x 3 x !cir.float>, !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>

// LLVM: %[[PTR_ADDR:.*]] = alloca ptr, align 8
// LLVM: %[[MATRIX_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[TMP_PTR:.*]] = load ptr, ptr %[[PTR_ADDR]], align 8
// LLVM: %[[RESULT:.*]] = call <9 x float> @llvm.matrix.column.major.load.v9f32.i64(ptr align 4 %[[TMP_PTR]], i64 3, i1 true, i32 3, i32 3)
// LLVM: store <9 x float> %[[RESULT]], ptr %[[MATRIX_ADDR]], align 4

void column_major_store() {
  matrix3x3 matrix;
  float *ptr;
  __builtin_matrix_column_major_store(matrix, ptr, 3);
}

// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[PTR_ADDR:.*]] = cir.alloca "ptr" {{.*}} : !cir.ptr<!cir.ptr<!cir.float>>
// CIR: %[[TMP_MATRIX:.*]] = cir.load {{.*}} %[[MATRIX_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: %[[TMP_PTR:.*]] = cir.load {{.*}} %[[PTR_ADDR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CIR: %[[STRIDE:.*]] = cir.const #cir.int<3> : !u64i
// CIR: cir.matrix.column_major_store %[[TMP_MATRIX]] : <3 x 3 x !cir.float>, %[[TMP_PTR]] : <!cir.float>, %[[STRIDE]] : !u64i

// LLVM: %[[MATRIX_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[PTR_ADDR:.*]] = alloca ptr, align 8
// LLVM: %[[TMP_MATRIX:.*]] = load <9 x float>, ptr %[[MATRIX_ADDR]], align 4
// LLVM: %[[TMP_PTR:.*]] = load ptr, ptr %[[PTR_ADDR]], align 8
// LLVM: call void @llvm.matrix.column.major.store.v9f32.i64(<9 x float> %[[TMP_MATRIX]], ptr align 4 %[[TMP_PTR]], i64 3, i1 false, i32 3, i32 3)

void column_major_volatile_store() {
  matrix3x3 matrix;
  volatile float *ptr;
  __builtin_matrix_column_major_store(matrix, ptr, 3);
}

// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[PTR_ADDR:.*]] = cir.alloca "ptr" {{.*}} : !cir.ptr<!cir.ptr<!cir.float>>
// CIR: %[[TMP_MATRIX:.*]] = cir.load {{.*}} %[[MATRIX_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: %[[TMP_PTR:.*]] = cir.load {{.*}} %[[PTR_ADDR]] : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
// CIR: %[[STRIDE:.*]] = cir.const #cir.int<3> : !u64i
// CIR: cir.matrix.column_major_store %[[TMP_MATRIX]] : <3 x 3 x !cir.float>, %[[TMP_PTR]] : <!cir.float>, %[[STRIDE]] : !u64i volatile

// LLVM: %[[MATRIX_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[PTR_ADDR:.*]] = alloca ptr, align 8
// LLVM: %[[TMP_MATRIX:.*]] = load <9 x float>, ptr %[[MATRIX_ADDR]], align 4
// LLVM: %[[TMP_PTR:.*]] = load ptr, ptr %[[PTR_ADDR]], align 8
// LLVM: call void @llvm.matrix.column.major.store.v9f32.i64(<9 x float> %[[TMP_MATRIX]], ptr align 4 %[[TMP_PTR]], i64 3, i1 true, i32 3, i32 3)

void matrix_subscript_expr() {
  matrix3x3 matrix;
  float b = matrix[1][2];
}

// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.float>
// CIR: %[[ROW_IDX:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[COLUMN_IDX:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[TMP_MATRIX:.*]] = cir.load {{.*}} %[[MATRIX_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: %[[ELEM:.*]] = cir.matrix.extract %[[TMP_MATRIX]][%[[ROW_IDX]] : !s32i] [%[[COLUMN_IDX]] : !s32i] : !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[ELEM]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[MATRIX_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, align 4
// LLVM: %[[TMP_MATRIX:.*]] = load <9 x float>, ptr %[[MATRIX_ADDR]], align 4
// LLVM: %[[ELEM:.*]] = extractelement <9 x float> %[[TMP_MATRIX]], i64 7
// LLVM: store float %[[ELEM]], ptr %[[B_ADDR]], align 4

void matrix_subscript_expr_non_const_indices() {
  matrix3x3 matrix;
  unsigned long row;
  unsigned long column;
  float b = matrix[row][column];
}

// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[ROW_ADDR:.*]] = cir.alloca "row" {{.*}} : !cir.ptr<!u64i>
// CIR: %[[COLUMN_ADDR:.*]] = cir.alloca "column" {{.*}} : !cir.ptr<!u64i>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.float>
// CIR: %[[ROW_IDX:.*]] = cir.load {{.*}} %[[ROW_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[COLUMN_IDX:.*]] = cir.load {{.*}} %[[COLUMN_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[TMP_MATRIX:.*]] = cir.load {{.*}} %[[MATRIX_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: %[[ELEM:.*]] = cir.matrix.extract %[[TMP_MATRIX]][%[[ROW_IDX]] : !u64i] [%[[COLUMN_IDX]] : !u64i] : !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[ELEM]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// LLVM: %[[MATRIX_ADDR:.*]] = alloca [9 x float], align 4
// LLVM: %[[ROW_ADDR:.*]] = alloca i64, align 8
// LLVM: %[[COLUMN_ADDR:.*]] = alloca i64, align 8
// LLVM: %[[B_ADDR:.*]] = alloca float, align 4
// LLVM: %[[ROW_IDX:.*]] = load i64, ptr %[[ROW_ADDR]], align 8
// LLVM: %[[COLUMN_IDX:.*]] = load i64, ptr %[[COLUMN_ADDR]], align 8

// Difference here because CIR calculate the flat index in lowering pass after matrix is loaded

// LLVM-CIR: %[[TMP_MATRIX:.*]] = load <9 x float>, ptr %[[MATRIX_ADDR]], align 4
// LLVM-CIR: %[[MUL_COL_3:.*]] = mul i64 %[[COLUMN_IDX]], 3
// LLVM-CIR: %[[FLAT_IDX:.*]] = add i64 %[[MUL_COL_3]], %[[ROW_IDX]]

// LLVM-OGCG: %[[MUL_COL_3:.*]] = mul i64 %[[COLUMN_IDX]], 3
// LLVM-OGCG: %[[FLAT_IDX:.*]] = add i64 %[[MUL_COL_3]], %[[ROW_IDX]]
// LLVM-OGCG: %[[TMP_MATRIX:.*]] = load <9 x float>, ptr %[[MATRIX_ADDR]], align 4

// LLVM: %[[ELEM:.*]] = extractelement <9 x float> %[[TMP_MATRIX]], i64 %[[FLAT_IDX]]
// LLVM: store float %[[ELEM]], ptr %[[B_ADDR]], align 4
