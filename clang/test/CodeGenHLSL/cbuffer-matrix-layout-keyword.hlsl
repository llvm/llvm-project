// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=row-major -o - | FileCheck %s

// row_major float2x3 -> 2 rows of <3 x float>, 16-byte row stride:
//   { [1 x { <3 x float>, pad(4) }], <3 x float> }
// CHECK: %__cblayout_CB_RM = type <{ <{ [1 x <{ <3 x float>, target("dx.Padding", 4) }>], <3 x float> }> }>

// column_major float2x3 -> 3 columns of <2 x float>, 16-byte column stride:
//   { [2 x { <2 x float>, pad(8) }], <2 x float> }
// CHECK: %__cblayout_CB_CM = type <{ <{ [2 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }> }>

// CHECK: @CB_RM.cb = global target("dx.CBuffer", %__cblayout_CB_RM)
// CHECK: @rm = external hidden addrspace(2) global <{ [1 x <{ <3 x float>, target("dx.Padding", 4) }>], <3 x float> }>, align 4
// CHECK: @CB_CM.cb = global target("dx.CBuffer", %__cblayout_CB_CM)
// CHECK: @cm = external hidden addrspace(2) global <{ [2 x <{ <2 x float>, target("dx.Padding", 8) }>], <2 x float> }>, align 4

cbuffer CB_RM {
  row_major float2x3 rm;
}
cbuffer CB_CM {
  column_major float2x3 cm;
}

RWBuffer<float> Out;

[numthreads(1,1,1)]
void main() {
  Out[0] = rm[0][0] + cm[0][0];
}

// row_major copy: dest is [2 x <3 x float>]; load two <3 x float> rows at +0/+16.
// CHECK-LABEL: define internal void @_Z4mainv()
// CHECK: %matrix.buf.copy = alloca [2 x <3 x float>], align 4
// CHECK: %matrix.buf.copy2 = alloca [3 x <2 x float>], align 4
// CHECK: load <3 x float>, ptr addrspace(2) @rm, align 4
// CHECK: load <3 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @rm, i32 16), align 4

// column_major copy: dest is [3 x <2 x float>]; load three <2 x float> columns at +0/+16/+32.
// CHECK: load <2 x float>, ptr addrspace(2) @cm, align 4
// CHECK: load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @cm, i32 16), align 4
// CHECK: load <2 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @cm, i32 32), align 4
