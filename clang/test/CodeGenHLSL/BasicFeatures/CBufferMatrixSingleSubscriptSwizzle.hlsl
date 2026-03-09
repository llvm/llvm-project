// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -disable-llvm-passes -emit-llvm -finclude-default-header -fmatrix-memory-layout=row-major -o - %s | FileCheck %s

cbuffer CB {
  float3x2 Mat;
}

// CHECK: @Mat = external hidden addrspace(2) global <{ [2 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>, align 4

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> @_Z15get_row_swizzlev()
// CHECK: %matrix.buf.copy = alloca [3 x <2 x float>], align 8
// CHECK: %cbuf.load = load <2 x float>, ptr addrspace(2) @Mat, align 8
// CHECK: %cbuf.load2 = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 16), align 8
// CHECK: %cbuf.load4 = load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @Mat, i32 32), align 8
// CHECK: %0 = load <6 x float>, ptr %matrix.buf.copy, align 8
// CHECK: %1 = shufflevector <6 x float> %0, <6 x float> poison, <2 x i32> <i32 4, i32 1>

float2 get_row_swizzle() {
  return Mat[1].gr;
}
