// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -fmatrix-memory-layout=column-major -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -fmatrix-memory-layout=row-major -o - | FileCheck %s

// Verifies that a per-decl `[[hlsl::row_major]]` / `[[hlsl::column_major]]`
// (spelled `row_major` / `column_major` in HLSL) overrides the
// `-fmatrix-memory-layout=` default at every CodeGen lowering site:
//
//   * `MatrixSubscriptExpr` index computation
//   * `MatrixSingleSubscriptExpr` row extraction
//   * `__builtin_hlsl_mul` matrix-multiply transpose insertion
//   * `__builtin_hlsl_transpose` row/col dimension swap
//   * `CK_HLSLMatrixTruncation` shuffle mask
//
// The decl-level attribute should win regardless of the TU default.

// -----------------------------------------------------------------------------
// MatrixSubscriptExpr indexing: row-major attr -> Row*NumCols + Col
// -----------------------------------------------------------------------------
export float subscript_rm(int row, int col, row_major float2x3 m) {
  return m[row][col];
}
// CHECK-LABEL: define {{.*}} float @_Z12subscript_rmiiu11matrix_typeILm2ELm3EfE
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: [[COL:%.*]] = load i32, ptr %col.addr
// CHECK: [[OFFSET:%.*]] = mul i32 [[ROW]], 3
// CHECK: [[IDX:%.*]] = add i32 [[OFFSET]], [[COL]]
// CHECK: extractelement <6 x float> %{{.*}}, i32 [[IDX]]

// -----------------------------------------------------------------------------
// MatrixSubscriptExpr indexing: column-major attr -> Col*NumRows + Row
// -----------------------------------------------------------------------------
export float subscript_cm(int row, int col, column_major float2x3 m) {
  return m[row][col];
}
// CHECK-LABEL: define {{.*}} float @_Z12subscript_cmiiu11matrix_typeILm2ELm3EfE
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: [[COL:%.*]] = load i32, ptr %col.addr
// CHECK: [[OFFSET:%.*]] = mul i32 [[COL]], 2
// CHECK: [[IDX:%.*]] = add i32 [[OFFSET]], [[ROW]]
// CHECK: extractelement <6 x float> %{{.*}}, i32 [[IDX]]

// -----------------------------------------------------------------------------
// MatrixSingleSubscriptExpr (row extraction): attribute selects the per-element
// index formula even when the TU default disagrees.
// -----------------------------------------------------------------------------

// Row-major: per-column element index is Row*NumCols + Col, materialized as a
// constant-zero / constant-one / constant-two add to (Row*3).
export float3 row_extract_rm(int row, row_major float2x3 m) {
  return m[row];
}
// CHECK-LABEL: define {{.*}} <3 x float> @_Z14row_extract_rmiu11matrix_typeILm2ELm3EfE
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: [[ROW_OFFSET0:%.*]] = mul i32 [[ROW]], 3
// CHECK: add i32 [[ROW_OFFSET0]], 0
// CHECK: [[ROW_OFFSET1:%.*]] = mul i32 [[ROW]], 3
// CHECK: add i32 [[ROW_OFFSET1]], 1
// CHECK: [[ROW_OFFSET2:%.*]] = mul i32 [[ROW]], 3
// CHECK: add i32 [[ROW_OFFSET2]], 2

// Column-major: per-column element index is Col*NumRows + Row, so we *don't*
// see the Row*NumCols multiply; instead each column folds the constant
// Col*NumRows into the GEP, leaving just an add of Row.
export float3 row_extract_cm(int row, column_major float2x3 m) {
  return m[row];
}
// CHECK-LABEL: define {{.*}} <3 x float> @_Z14row_extract_cmiu11matrix_typeILm2ELm3EfE
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: add i32 0, [[ROW]]
// CHECK: add i32 2, [[ROW]]
// CHECK: add i32 4, [[ROW]]

// -----------------------------------------------------------------------------
// __builtin_hlsl_mul (vector * matrix): row-major operand triggers a transpose
// before the column-major matrix.multiply intrinsic.
// -----------------------------------------------------------------------------
export float3 vec_mat_rm(float2 v, row_major float2x3 m) { return mul(v, m); }
// CHECK-LABEL: define {{.*}} <3 x float> @_Z10vec_mat_rmDv2_fu11matrix_typeILm2ELm3EfE
// CHECK: [[T:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)
// CHECK: call {{.*}} <3 x float> @llvm.matrix.multiply.v3f32.v2f32.v6f32(<2 x float> %{{.*}}, <6 x float> [[T]], i32 1, i32 2, i32 3)

// Column-major operand: no transpose is inserted before matrix.multiply.
export float3 vec_mat_cm(float2 v, column_major float2x3 m) { return mul(v, m); }
// CHECK-LABEL: define {{.*}} <3 x float> @_Z10vec_mat_cmDv2_fu11matrix_typeILm2ELm3EfE
// CHECK-NOT: @llvm.matrix.transpose
// CHECK: call {{.*}} <3 x float> @llvm.matrix.multiply.v3f32.v2f32.v6f32(<2 x float> %{{.*}}, <6 x float> %{{.*}}, i32 1, i32 2, i32 3)

// -----------------------------------------------------------------------------
// __builtin_hlsl_transpose: row-major operand swaps Rows/Cols passed to the
// underlying intrinsic.
// -----------------------------------------------------------------------------

// Row-major float2x3 transposed: passes (Cols=3, Rows=2) to the intrinsic.
export float3x2 transpose_rm(row_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z12transpose_rmu11matrix_typeILm2ELm3EfE
// CHECK: call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)

// Column-major float2x3 transposed: passes (Rows=2, Cols=3) to the intrinsic.
export float3x2 transpose_cm(column_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z12transpose_cmu11matrix_typeILm2ELm3EfE
// CHECK: call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 2, i32 3)

// -----------------------------------------------------------------------------
// CK_HLSLMatrixTruncation: the shuffle mask that picks elements from the
// source matrix uses the operand's per-decl layout to flatten indices.
// -----------------------------------------------------------------------------

// Row-major source 3x2 -> row-major dest 2x2: flat row-major mask is {0,1,2,3}.
export float2x2 truncate_rm(row_major float3x2 m) { return (float2x2)m; }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z11truncate_rmu11matrix_typeILm3ELm2EfE
// CHECK: shufflevector <6 x float> %{{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

// Column-major source 3x2 -> column-major dest 2x2: flat column-major mask is {0,1,3,4}.
export float2x2 truncate_cm(column_major float3x2 m) { return (float2x2)m; }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z11truncate_cmu11matrix_typeILm3ELm2EfE
// CHECK: shufflevector <6 x float> %{{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 1, i32 3, i32 4>

// -----------------------------------------------------------------------------
// Array of matrix: the per-decl layout attribute propagates through
// ConstantArrayType sugar via wrapMatrixWithLayoutAttr, so indexing into an
// array element still uses the correct layout.
// -----------------------------------------------------------------------------

// Row-major array element subscript: Row*NumCols + Col
export float arr_subscript_rm(int row, int col, row_major float2x3 arr[2]) {
  return arr[1][row][col];
}
// CHECK-LABEL: define {{.*}} float @_Z16arr_subscript_rm
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: [[COL:%.*]] = load i32, ptr %col.addr
// CHECK: [[OFFSET:%.*]] = mul i32 [[ROW]], 3
// CHECK: [[IDX:%.*]] = add i32 [[OFFSET]], [[COL]]
// CHECK: extractelement <6 x float> %{{.*}}, i32 [[IDX]]

// Column-major array element subscript: Col*NumRows + Row
export float arr_subscript_cm(int row, int col, column_major float2x3 arr[2]) {
  return arr[1][row][col];
}
// CHECK-LABEL: define {{.*}} float @_Z16arr_subscript_cm
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: [[COL:%.*]] = load i32, ptr %col.addr
// CHECK: [[OFFSET:%.*]] = mul i32 [[COL]], 2
// CHECK: [[IDX:%.*]] = add i32 [[OFFSET]], [[ROW]]
// CHECK: extractelement <6 x float> %{{.*}}, i32 [[IDX]]

// -----------------------------------------------------------------------------
// Multi-dimensional array of matrix: wrapMatrixWithLayoutAttr recurses
// through nested ConstantArrayType layers.
// -----------------------------------------------------------------------------

// Row-major 2D array element subscript: Row*NumCols + Col
export float arr2d_subscript_rm(int row, int col, row_major float2x3 arr[2][3]) {
  return arr[0][1][row][col];
}
// CHECK-LABEL: define {{.*}} float @_Z18arr2d_subscript_rm
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: [[COL:%.*]] = load i32, ptr %col.addr
// CHECK: [[OFFSET:%.*]] = mul i32 [[ROW]], 3
// CHECK: [[IDX:%.*]] = add i32 [[OFFSET]], [[COL]]
// CHECK: extractelement <6 x float> %{{.*}}, i32 [[IDX]]

// Column-major 2D array element subscript: Col*NumRows + Row
export float arr2d_subscript_cm(int row, int col, column_major float2x3 arr[2][3]) {
  return arr[0][1][row][col];
}
// CHECK-LABEL: define {{.*}} float @_Z18arr2d_subscript_cm
// CHECK: [[ROW:%.*]] = load i32, ptr %row.addr
// CHECK: [[COL:%.*]] = load i32, ptr %col.addr
// CHECK: [[OFFSET:%.*]] = mul i32 [[COL]], 2
// CHECK: [[IDX:%.*]] = add i32 [[OFFSET]], [[ROW]]
// CHECK: extractelement <6 x float> %{{.*}}, i32 [[IDX]]
