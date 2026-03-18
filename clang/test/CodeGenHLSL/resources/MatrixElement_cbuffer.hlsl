// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -disable-llvm-passes \
// RUN:   -emit-llvm -finclude-default-header -fmatrix-memory-layout=column-major \
// RUN:   -o - %s | FileCheck %s --check-prefix=CHECK,COL-CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -disable-llvm-passes \
// RUN:   -emit-llvm -finclude-default-header -fmatrix-memory-layout=row-major \
// RUN:   -o - %s | FileCheck %s --check-prefixes=CHECK,ROW-CHECK


cbuffer CB {
  float3x2 Mat;
};


// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float @_Z23getCBufferScalarElementv(
// CHECK-SAME: ) #[[ATTR2:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// COL-CHECK-NEXT:    [[MATRIX_BUF_COPY:%.*]] = alloca [2 x <3 x float>], align 4
// COL-CHECK-NEXT:    [[CBUF_DEST:%.*]] = getelementptr inbounds [2 x <3 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 0
// COL-CHECK-NEXT:    [[CBUF_LOAD:%.*]] = load <3 x float>, ptr addrspace(2) @Mat, align 4
// COL-CHECK-NEXT:    store <3 x float> [[CBUF_LOAD]], ptr [[CBUF_DEST]], align 4
// COL-CHECK-NEXT:    [[CBUF_DEST1:%.*]] = getelementptr inbounds [2 x <3 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 1
// COL-CHECK-NEXT:    [[CBUF_LOAD2:%.*]] = load <3 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 16), align 4
// COL-CHECK-NEXT:    store <3 x float> [[CBUF_LOAD2]], ptr [[CBUF_DEST1]], align 4
// COL-CHECK-NEXT:    [[TMP0:%.*]] = load <6 x float>, ptr [[MATRIX_BUF_COPY]], align 4
// COL-CHECK-NEXT:    [[TMP1:%.*]] = extractelement <6 x float> [[TMP0]], i32 4
// COL-CHECK-NEXT:    ret float [[TMP1]]
//
// ROW-CHECK-NEXT:    [[MATRIX_BUF_COPY:%.*]] = alloca [3 x <2 x float>], align 4
// ROW-CHECK-NEXT:    [[CBUF_DEST:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 0
// ROW-CHECK-NEXT:    [[CBUF_LOAD:%.*]] = load <2 x float>, ptr addrspace(2) @Mat, align 4
// ROW-CHECK-NEXT:    store <2 x float> [[CBUF_LOAD]], ptr [[CBUF_DEST]], align 4
// ROW-CHECK-NEXT:    [[CBUF_DEST1:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 1
// ROW-CHECK-NEXT:    [[CBUF_LOAD2:%.*]] = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 16), align 4
// ROW-CHECK-NEXT:    store <2 x float> [[CBUF_LOAD2]], ptr [[CBUF_DEST1]], align 4
// ROW-CHECK-NEXT:    [[CBUF_DEST3:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 2
// ROW-CHECK-NEXT:    [[CBUF_LOAD4:%.*]] = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 32), align 4
// ROW-CHECK-NEXT:    store <2 x float> [[CBUF_LOAD4]], ptr [[CBUF_DEST3]], align 4
// ROW-CHECK-NEXT:    [[TMP0:%.*]] = load <6 x float>, ptr [[MATRIX_BUF_COPY]], align 4
// ROW-CHECK-NEXT:    [[TMP1:%.*]] = extractelement <6 x float> [[TMP0]], i32 3
// ROW-CHECK-NEXT:    ret float [[TMP1]]
//
float getCBufferScalarElement() {
  return Mat._22;
}


// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> @_Z23getCBufferSwizzleAccessv(
// CHECK-SAME: ) #[[ATTR2]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// COL-CHECK-NEXT:    [[MATRIX_BUF_COPY:%.*]] = alloca [2 x <3 x float>], align 4
// COL-CHECK-NEXT:    [[CBUF_DEST:%.*]] = getelementptr inbounds [2 x <3 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 0
// COL-CHECK-NEXT:    [[CBUF_LOAD:%.*]] = load <3 x float>, ptr addrspace(2) @Mat, align 4
// COL-CHECK-NEXT:    store <3 x float> [[CBUF_LOAD]], ptr [[CBUF_DEST]], align 4
// COL-CHECK-NEXT:    [[CBUF_DEST1:%.*]] = getelementptr inbounds [2 x <3 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 1
// COL-CHECK-NEXT:    [[CBUF_LOAD2:%.*]] = load <3 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 16), align 4
// COL-CHECK-NEXT:    store <3 x float> [[CBUF_LOAD2]], ptr [[CBUF_DEST1]], align 4
// COL-CHECK-NEXT:    [[TMP0:%.*]] = load <6 x float>, ptr [[MATRIX_BUF_COPY]], align 4
// COL-CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <6 x float> [[TMP0]], <6 x float> poison, <4 x i32> <i32 0, i32 3, i32 1, i32 4>
// COL-CHECK-NEXT:    ret <4 x float> [[TMP1]]
//
// ROW-CHECK-NEXT:    [[MATRIX_BUF_COPY:%.*]] = alloca [3 x <2 x float>], align 4
// ROW-CHECK-NEXT:    [[CBUF_DEST:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 0
// ROW-CHECK-NEXT:    [[CBUF_LOAD:%.*]] = load <2 x float>, ptr addrspace(2) @Mat, align 4
// ROW-CHECK-NEXT:    store <2 x float> [[CBUF_LOAD]], ptr [[CBUF_DEST]], align 4
// ROW-CHECK-NEXT:    [[CBUF_DEST1:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 1
// ROW-CHECK-NEXT:    [[CBUF_LOAD2:%.*]] = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 16), align 4
// ROW-CHECK-NEXT:    store <2 x float> [[CBUF_LOAD2]], ptr [[CBUF_DEST1]], align 4
// ROW-CHECK-NEXT:    [[CBUF_DEST3:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 2
// ROW-CHECK-NEXT:    [[CBUF_LOAD4:%.*]] = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 32), align 4
// ROW-CHECK-NEXT:    store <2 x float> [[CBUF_LOAD4]], ptr [[CBUF_DEST3]], align 4
// ROW-CHECK-NEXT:    [[TMP0:%.*]] = load <6 x float>, ptr [[MATRIX_BUF_COPY]], align 4
// ROW-CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <6 x float> [[TMP0]], <6 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// ROW-CHECK-NEXT:    ret <4 x float> [[TMP1]]
//
float4 getCBufferSwizzleAccess() {
  return Mat._11_12_21_22;
}

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z22getZeroBasedSwizzleEltu11matrix_typeILm3ELm2EfE(
// CHECK-SAME: <6 x float> noundef nofpclass(nan inf) [[M:%.*]]) #[[ATTR2]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// COL-CHECK-NEXT:    [[M_ADDR:%.*]] = alloca [2 x <3 x float>], align 4
// COL-CHECK-NEXT:    store <6 x float> [[M]], ptr [[M_ADDR]], align 4
// COL-CHECK-NEXT:    [[TMP0:%.*]] = load <6 x float>, ptr [[M_ADDR]], align 4
// COL-CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <6 x float> [[TMP0]], <6 x float> poison, <2 x i32> <i32 1, i32 3>
// COL-CHECK-NEXT:    ret <2 x float> [[TMP1]]
//
// ROW-CHECK-NEXT:    [[M_ADDR:%.*]] = alloca [3 x <2 x float>], align 4
// ROW-CHECK-NEXT:    store <6 x float> [[M]], ptr [[M_ADDR]], align 4
// ROW-CHECK-NEXT:    [[TMP0:%.*]] = load <6 x float>, ptr [[M_ADDR]], align 4
// ROW-CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <6 x float> [[TMP0]], <6 x float> poison, <2 x i32> <i32 2, i32 1>
// ROW-CHECK-NEXT:    ret <2 x float> [[TMP1]]
//
float2 getZeroBasedSwizzleElt(float3x2 M) {
  return M._m10_m01;
}
