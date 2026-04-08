// RUN: %clang_cc1 -fenable-matrix  -fmatrix-memory-layout=column-major -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - -std=c++11 | FileCheck %s

 typedef float fx2x3_t __attribute__((matrix_type(2, 3)));
 float Out[6];

void binaryOpMatrixSubscriptExpr(int index, fx2x3_t M) {
    // CHECK-LABEL: binaryOpMatrixSubscriptExpr
    // CHECK: %row = alloca i32, align 4
    // CHECK: %col = alloca i32, align 4
    // CHECK: [[row_load:%.*]] = load i32, ptr %row, align 4
    // CHECK-NEXT: [[row_load_zext:%.*]] = zext i32 [[row_load]] to i64
    // CHECK-NEXT: [[col_load:%.*]] = load i32, ptr %col, align 4
    // CHECK-NEXT: [[col_load_zext:%.*]] = zext i32 [[col_load]] to i64
    // CHECK-NEXT: [[col_offset:%.*]] = mul i64 [[col_load_zext]], 2
    // CHECK-NEXT: [[col_major_index:%.*]] = add i64 [[col_offset]], [[row_load_zext]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x float>, ptr %M.addr, align 4
    // CHECK-NEXT: %matrixext = extractelement <6 x float> [[matrix_as_vec]], i64 [[col_major_index]]
    const unsigned int COLS = 3;
    unsigned int row = index / COLS;
    unsigned int col = index % COLS;
    Out[index] = M[row][col];
}

float returnMatrixSubscriptExpr(int row, int col, fx2x3_t M) {
    // CHECK-LABEL: returnMatrixSubscriptExpr
    // CHECK: [[row_load:%.*]] = load i32, ptr %row.addr, align 4
    // CHECK-NEXT: [[row_load_sext:%.*]] = sext i32 [[row_load]] to i64
    // CHECK-NEXT: [[col_load:%.*]] = load i32, ptr %col.addr, align 4
    // CHECK-NEXT: [[col_load_sext:%.*]] = sext i32 [[col_load]] to i64
    // CHECK-NEXT: [[col_offset:%.*]] = mul i64 [[col_load_sext]], 2
    // CHECK-NEXT: [[col_major_index:%.*]] = add i64 [[col_offset]], [[row_load_sext]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x float>, ptr %M.addr, align 4
    // CHECK-NEXT: %matrixext = extractelement <6 x float> [[matrix_as_vec]], i64 [[col_major_index]]
    return M[row][col];
}
