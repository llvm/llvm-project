// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// NOTE: This test is only to confirm we can do codgen with the matrix alias.

RWBuffer<int> Out : register(u1);

void one_index_test(int index, half2x3 M) {
    // CHECK-LABEL: one_index_test
    // CHECK: %row = alloca i32, align 4
    // CHECK: %col = alloca i32, align 4
    // CHECK: [[row_load:%.*]] = load i32, ptr %row, align 4
    // CHECK-NEXT: [[col_load:%.*]] = load i32, ptr %col, align 4
    // CHECK-NEXT: [[row_offset:%.*]] = mul i32 [[row_load]], 3
    // CHECK-NEXT: [[row_major_index:%.*]] = add i32 [[row_offset]], [[col_load]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x half>, ptr %M.addr, align 2
    // CHECK-NEXT: %matrixext = extractelement <6 x half> [[matrix_as_vec]], i32 [[row_major_index]]
    const uint COLS = 3;
    uint row = index / COLS;
    uint col = index % COLS;
    Out[index] = M[row][col];
}

half two_index_test(int row, int col, half2x3 M) {
    // CHECK-LABEL: two_index_test
    // CHECK: [[row_offset:%.*]] = mul i32 [[row_load:%.*]], 3
    // CHECK-NEXT: [[row_major_index:%.*]] = add i32 [[row_offset]], [[col_load:%.*]]
    // CHECK-NEXT: [[matrix_as_vec:%.*]] = load <6 x half>, ptr %M.addr, align 2
    // CHECK-NEXT: %matrixext = extractelement <6 x half> [[matrix_as_vec]], i32 [[row_major_index]]
    return M[row][col];
}
