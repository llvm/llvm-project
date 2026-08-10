// RUN: %clang_cc1 -fenable-matrix -fmatrix-memory-layout=row-major -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,ROW-CHECK
// RUN: %clang_cc1 -fenable-matrix -fmatrix-memory-layout=column-major -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,COL-CHECK
// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,COL-CHECK

typedef float fx2x3_t __attribute__((matrix_type(2, 3)));
 float Out[6];

 fx2x3_t gM;

void binaryOpMatrixSubscriptExpr(int index, fx2x3_t M) {
    // CHECK-LABEL: binaryOpMatrixSubscriptExpr
    // CHECK: %row = alloca i32, align 4
    // CHECK: %col = alloca i32, align 4
    // CHECK: [[row_load:%.*]] = load i32, ptr %row, align 4
    // CHECK-NEXT: [[row_load_zext:%.*]] = zext i32 [[row_load]] to i64
    // CHECK-NEXT: [[col_load:%.*]] = load i32, ptr %col, align 4
    // CHECK-NEXT: [[col_load_zext:%.*]] = zext i32 [[col_load]] to i64
    // COL-CHECK-NEXT: [[col_offset:%.*]] = mul i64 [[col_load_zext]], 2
    // COL-CHECK-NEXT: [[col_major_index:%.*]] = add i64 [[col_offset]], [[row_load_zext]]
    // ROW-CHECK-NEXT: [[row_offset:%.*]] = mul i64 [[row_load_zext]], 3
    // ROW-CHECK-NEXT: [[row_major_index:%.*]] = add i64 [[row_offset]], [[col_load_zext]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x float>, ptr %M.addr, align 4
    // COL-CHECK-NEXT: %matrixext = extractelement <6 x float> [[matrix_as_vec]], i64 [[col_major_index]]
    // ROW-CHECK-NEXT: %matrixext = extractelement <6 x float> [[matrix_as_vec]], i64 [[row_major_index]]
    const unsigned int COLS = 3;
    unsigned int row = index / COLS;
    unsigned int col = index % COLS;
    Out[index] = M[row][col];
}

float returnMatrixSubscriptExpr(int row, int col, fx2x3_t M) {
    // CHECK-LABEL: returnMatrixSubscriptExpr
    // CHECK: [[row_load:%.*]] = load i32, ptr [[row_ptr:%.*]], align 4
    // CHECK-NEXT: [[row_load_sext:%.*]] = sext i32 [[row_load]] to i64
    // CHECK-NEXT: [[col_load:%.*]] = load i32, ptr [[col_ptr:%.*]], align 4
    // CHECK-NEXT: [[col_load_sext:%.*]] = sext i32 [[col_load]] to i64
    // COL-CHECK-NEXT: [[col_offset:%.*]] = mul i64 [[col_load_sext]], 2
    // COL-CHECK-NEXT: [[col_major_index:%.*]] = add i64 [[col_offset]], [[row_load_sext]]
    // ROW-CHECK-NEXT: [[row_offset:%.*]] = mul i64 [[row_load_sext]], 3
    // ROW-CHECK-NEXT: [[row_major_index:%.*]] = add i64 [[row_offset]], [[col_load_sext]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x float>, ptr %M.addr, align 4
    // COL-CHECK-NEXT: [[matrix_after_extract:%.*]] = extractelement <6 x float> [[matrix_as_vec]], i64 [[col_major_index]]
    // ROW-CHECK-NEXT: [[matrix_after_extract:%.*]] = extractelement <6 x float> [[matrix_as_vec]], i64 [[row_major_index]]
    // CHECK-NEXT: ret float [[matrix_after_extract]]
    return M[row][col];
}

void storeAtMatrixSubscriptExpr(int row, int col, float value) {
    // CHECK-LABEL: storeAtMatrixSubscriptExpr
    // CHECK: [[value_load:%.*]] = load float, ptr [[value_ptr:%.*]], align 4
    // ROW-CHECK: [[row_offset:%.*]] = mul i64 [[row_load:%.*]], 3
    // ROW-CHECK-NEXT: [[row_major_index:%.*]] = add i64 [[row_offset]], [[col_load:%.*]]
    // COL-CHECK: [[col_offset:%.*]] = mul i64 [[col_load:%.*]], 2
    // COL-CHECK-NEXT: [[col_major_index:%.*]] = add i64 [[col_offset]], [[row_load:%.*]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x float>, ptr @gM, align 4
    // ROW-CHECK-NEXT: [[matrix_after_insert:%.*]] = insertelement <6 x float> [[matrix_as_vec]], float [[value_load]], i64 [[row_major_index]]
    // COL-CHECK-NEXT: [[matrix_after_insert:%.*]] = insertelement <6 x float> [[matrix_as_vec]], float [[value_load]], i64 [[col_major_index]]
    // CHECK-NEXT: store <6 x float> [[matrix_after_insert]], ptr @gM, align 4
    gM[row][col] = value;
}
