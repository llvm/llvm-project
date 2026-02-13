// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=row-major -o - | FileCheck %s --check-prefixes=CHECK,ROW-CHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - | FileCheck %s --check-prefixes=CHECK,COL-CHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,COL-CHECK

RWBuffer<int> Out : register(u1);
half2x3 gM;


void binaryOpMatrixSubscriptExpr(int index, half2x3 M) {
    // CHECK-LABEL: binaryOpMatrixSubscriptExpr
    // CHECK: %row = alloca i32, align 4
    // CHECK: %col = alloca i32, align 4
    // CHECK: [[row_load:%.*]] = load i32, ptr %row, align 4
    // CHECK-NEXT: [[col_load:%.*]] = load i32, ptr %col, align 4
    // ROW-CHECK-NEXT: [[row_offset:%.*]] = mul i32 [[row_load]], 3
    // ROW-CHECK-NEXT: [[row_major_index:%.*]] = add i32 [[row_offset]], [[col_load]]
    // COL-CHECK-NEXT: [[col_offset:%.*]] = mul i32 [[col_load]], 2
    // COL-CHECK-NEXT: [[col_major_index:%.*]] = add i32 [[col_offset]], [[row_load]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x half>, ptr %M.addr, align 2
    // ROW-CHECK-NEXT: %matrixext = extractelement <6 x half> [[matrix_as_vec]], i32 [[row_major_index]]
    // COL-CHECK-NEXT: %matrixext = extractelement <6 x half> [[matrix_as_vec]], i32 [[col_major_index]]
    const uint COLS = 3;
    uint row = index / COLS;
    uint col = index % COLS;
    Out[index] = M[row][col];
}

half returnMatrixSubscriptExpr(int row, int col, half2x3 M) {
    // CHECK-LABEL: returnMatrixSubscriptExpr
    // ROW-CHECK: [[row_offset:%.*]] = mul i32 [[row_load:%.*]], 3
    // ROW-CHECK-NEXT: [[row_major_index:%.*]] = add i32 [[row_offset]], [[col_load:%.*]]
    // COL-CHECK: [[col_offset:%.*]] = mul i32 [[col_load:%.*]], 2
    // COL-CHECK-NEXT: [[col_major_index:%.*]] = add i32 [[col_offset]], [[row_load:%.*]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x half>, ptr %M.addr, align 2
    // ROW-CHECK-NEXT: %matrixext = extractelement <6 x half> [[matrix_as_vec]], i32 [[row_major_index]]
    // COL-CHECK-NEXT: %matrixext = extractelement <6 x half> [[matrix_as_vec]], i32 [[col_major_index]]
    return M[row][col];
}

void storeAtMatrixSubscriptExpr(int row, int col, half value) {
    // CHECK-LABEL: storeAtMatrixSubscriptExpr
    // CHECK: [[value_load:%.*]] = load half, ptr [[value_ptr:%.*]], align 2
    // ROW-CHECK: [[row_offset:%.*]] = mul i32 [[row_load:%.*]], 3
    // ROW-CHECK-NEXT: [[row_major_index:%.*]] = add i32 [[row_offset]], [[col_load:%.*]]
    // COL-CHECK: [[col_offset:%.*]] = mul i32 [[col_load:%.*]], 2
    // COL-CHECK-NEXT: [[col_major_index:%.*]] = add i32 [[col_offset]], [[row_load:%.*]]
    // ROW-CHECK-NEXT: [[matrix_gep:%.*]] = getelementptr <6 x half>, ptr addrspace(2) @gM, i32 0, i32 [[row_major_index]]
    // COL-CHECK-NEXT: [[matrix_gep:%.*]] = getelementptr <6 x half>, ptr addrspace(2) @gM, i32 0, i32 [[col_major_index]]
    // CHECK-NEXT: store half [[value_load]], ptr addrspace(2) [[matrix_gep]], align 2
    gM[row][col] = value;
}
