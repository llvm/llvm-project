// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -disable-llvm-passes -emit-llvm -finclude-default-header -fmatrix-memory-layout=row-major -o - %s | FileCheck %s

cbuffer CB {
  float3x2 Mat;
}

// CHECK: @Mat = external hidden addrspace(2) global <{ [2 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>, align 4

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z15get_row_swizzlev(
// CHECK-SAME: ) #[[ATTR2:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MATRIX_BUF_COPY:%.*]] = alloca [3 x <2 x float>], align 8
// CHECK-NEXT:    [[CBUF_DEST:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 0
// CHECK-NEXT:    [[CBUF_LOAD:%.*]] = load <2 x float>, ptr addrspace(2) @Mat, align 8
// CHECK-NEXT:    store <2 x float> [[CBUF_LOAD]], ptr [[CBUF_DEST]], align 8
// CHECK-NEXT:    [[CBUF_DEST1:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 1
// CHECK-NEXT:    [[CBUF_LOAD2:%.*]] = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 16), align 8
// CHECK-NEXT:    store <2 x float> [[CBUF_LOAD2]], ptr [[CBUF_DEST1]], align 8
// CHECK-NEXT:    [[CBUF_DEST3:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[MATRIX_BUF_COPY]], i32 0, i32 2
// CHECK-NEXT:    [[CBUF_LOAD4:%.*]] = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 32), align 8
// CHECK-NEXT:    store <2 x float> [[CBUF_LOAD4]], ptr [[CBUF_DEST3]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load <6 x float>, ptr [[MATRIX_BUF_COPY]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = shufflevector <6 x float> [[TMP0]], <6 x float> poison, <2 x i32> <i32 4, i32 1>
// CHECK-NEXT:    ret <2 x float> [[TMP1]]
//
float2 get_row_swizzle() {
  return Mat[1].gr;
}
