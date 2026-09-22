// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - | FileCheck %s --check-prefixes=CHECK,COLMAJOR
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=row-major -o - | FileCheck %s --check-prefixes=CHECK,ROWMAJOR

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
// __builtin_hlsl_mul (matrix * matrix): mixed per-decl layouts cause a
// transpose only on the row-major operand.
// -----------------------------------------------------------------------------

// LHS row-major, RHS column-major: only LHS is transposed.
export float2x2 mat_mat_rm_cm(row_major float2x3 a, column_major float3x2 b) { return mul(a, b); }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z13mat_mat_rm_cm
// CHECK: [[AMat:%.*]] = load <6 x float>, ptr %a.addr, align 4
// CHECK: [[BMat:%.*]] = load <6 x float>, ptr %b.addr, align 4
// CHECK: [[T:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[AMat]], i32 3, i32 2)
// CHECK: call {{.*}} <4 x float> @llvm.matrix.multiply.v4f32.v6f32.v6f32(<6 x float> [[T]], <6 x float> [[BMat]], i32 2, i32 3, i32 2)

// LHS column-major, RHS row-major: only RHS is transposed.
export float2x2 mat_mat_cm_rm(column_major float2x3 a, row_major float3x2 b) { return mul(a, b); }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z13mat_mat_cm_rm
// CHECK: [[AMat:%.*]] = load <6 x float>, ptr %a.addr, align 4
// CHECK: [[BMat:%.*]] = load <6 x float>, ptr %b.addr, align 4
// CHECK: [[T:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[BMat]], i32 2, i32 3)
// CHECK: call {{.*}} <4 x float> @llvm.matrix.multiply.v4f32.v6f32.v6f32(<6 x float> [[AMat]], <6 x float> [[T]], i32 2, i32 3, i32 2)

// Destination layout: the result is column-major, so no transpose is needed.
export column_major float2x2 mat_mat_dst_cm(column_major float2x3 a, column_major float3x2 b) { return mul(a, b); }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z14mat_mat_dst_cm
// CHECK: [[MUL:%.*]] = call {{.*}} <4 x float> @llvm.matrix.multiply.v4f32.v6f32.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}}, i32 2, i32 3, i32 2)
// CHECK-NOT: @llvm.matrix.transpose

// Destination layout: the result is row-major, so a transpose is needed.
export row_major float2x2 mat_mat_dst_rm(column_major float2x3 a, column_major float3x2 b) { return mul(a, b); }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z14mat_mat_dst_rm
// CHECK: [[MUL:%.*]] = call {{.*}} <4 x float> @llvm.matrix.multiply.v4f32.v6f32.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}}, i32 2, i32 3, i32 2)
// CHECK: call {{.*}} <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> [[MUL]], i32 2, i32 2)


// Row-major source -> column-major destination: bits already transposed, no-op.
export column_major float3x2 transpose_rm_to_cm(row_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z18transpose_rm_to_cmu11matrix_typeILm2ELm3EfE
// CHECK-NOT: @llvm.matrix.transpose
// CHECK: ret <6 x float>

// Column-major source -> row-major destination: bits already transposed, no-op.
export row_major float3x2 transpose_cm_to_rm(column_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z18transpose_cm_to_rmu11matrix_typeILm2ELm3EfE
// CHECK-NOT: @llvm.matrix.transpose
// CHECK: ret <6 x float>

// Row-major source -> row-major destination: real transpose, dims swapped.
export row_major float3x2 transpose_rm_to_rm(row_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z18transpose_rm_to_rmu11matrix_typeILm2ELm3EfE
// CHECK: call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)

// Column-major source -> column-major destination: real transpose, natural dims.
export column_major float3x2 transpose_cm_to_cm(column_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z18transpose_cm_to_cmu11matrix_typeILm2ELm3EfE
// CHECK: call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 2, i32 3)

// Default-layout return type: the TU `-fmatrix-memory-layout=` default 
// flips between a real transpose and a no-op depending on the default.
export float3x2 transpose_rm(row_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z12transpose_rmu11matrix_typeILm2ELm3EfE
// COLMAJOR-NOT: @llvm.matrix.transpose
// COLMAJOR: ret <6 x float>
// ROWMAJOR: call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 3, i32 2)


// column-major default: src/dst match -> real transpose, natural dims.
// row-major default: src/dst differ -> bits already transposed, no-op.
export float3x2 transpose_cm(column_major float2x3 m) { return transpose(m); }
// CHECK-LABEL: define {{.*}} <6 x float> @_Z12transpose_cmu11matrix_typeILm2ELm3EfE
// COLMAJOR: call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> %{{.*}}, i32 2, i32 3)
// ROWMAJOR-NOT: @llvm.matrix.transpose
// ROWMAJOR: ret <6 x float>

// -----------------------------------------------------------------------------
// CK_HLSLMatrixTruncation: the shuffle mask that picks elements from the
// source matrix uses the operand's per-decl layout to flatten indices.
// -----------------------------------------------------------------------------

typedef row_major    float2x2 RM22;
typedef column_major float2x2 CM22;
typedef row_major    float3x3 RM33;
typedef column_major float3x3 CM33;

// Row-major source 3x2 -> row-major dest 2x2: flat row-major mask is {0,1,2,3}.
export row_major float2x2 truncate_rm(row_major float3x2 m) { return (RM22)m; }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z11truncate_rmu11matrix_typeILm3ELm2EfE
// CHECK: shufflevector <6 x float> %{{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

// Column-major source 3x2 -> column-major dest 2x2: flat column-major mask is {0,1,3,4}.
export column_major float2x2 truncate_cm(column_major float3x2 m) { return (CM22)m; }
// CHECK-LABEL: define {{.*}} <4 x float> @_Z11truncate_cmu11matrix_typeILm3ELm2EfE
// CHECK: shufflevector <6 x float> %{{.*}}, <6 x float> poison, <4 x i32> <i32 0, i32 1, i32 3, i32 4>

// -----------------------------------------------------------------------------
// CK_HLSLMatrixTruncation cross-layout: when source and destination carry
// different layout keywords, `IsSrcRowMajor` and `IsDstRowMajor` differ. The
// source indices flatten using the source layout while the destination
// positions flatten using the destination layout. This is independent of the
// `-fmatrix-memory-layout=` default.
// -----------------------------------------------------------------------------

// Row-major src 3x4 -> column-major dst 3x3.
// src idx (R,C) = R*4+C; dst slot (R,C) = C*3+R.
//   (0,0)->mask[0]=0  (0,1)->mask[3]=1  (0,2)->mask[6]=2
//   (1,0)->mask[1]=4  (1,1)->mask[4]=5  (1,2)->mask[7]=6
//   (2,0)->mask[2]=8  (2,1)->mask[5]=9  (2,2)->mask[8]=10
export column_major float3x3 truncate_rm_to_cm(row_major float3x4 m) { return (CM33)m; }
// CHECK-LABEL: define {{.*}} <9 x float> @_Z17truncate_rm_to_cmu11matrix_typeILm3ELm4EfE
// CHECK: shufflevector <12 x float> %{{.*}}, <12 x float> poison, <9 x i32> <i32 0, i32 4, i32 8, i32 1, i32 5, i32 9, i32 2, i32 6, i32 10>

// Column-major src 3x4 -> row-major dst 3x3.
// src idx (R,C) = C*3+R; dst slot (R,C) = R*3+C.
//   (0,0)->mask[0]=0  (0,1)->mask[1]=3  (0,2)->mask[2]=6
//   (1,0)->mask[3]=1  (1,1)->mask[4]=4  (1,2)->mask[5]=7
//   (2,0)->mask[6]=2  (2,1)->mask[7]=5  (2,2)->mask[8]=8
export row_major float3x3 truncate_cm_to_rm(column_major float3x4 m) { return (RM33)m; }
// CHECK-LABEL: define {{.*}} <9 x float> @_Z17truncate_cm_to_rmu11matrix_typeILm3ELm4EfE
// CHECK: shufflevector <12 x float> %{{.*}}, <12 x float> poison, <9 x i32> <i32 0, i32 3, i32 6, i32 1, i32 4, i32 7, i32 2, i32 5, i32 8>

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
