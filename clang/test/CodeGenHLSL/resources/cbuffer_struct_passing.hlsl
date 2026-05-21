// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK-SPIR

struct S {
  float3 a;
  float4 b;
};
// CHECK-DXIL-DAG: %S = type <{ <3 x float>, target("dx.Padding", 4), <4 x float> }>
// CHECK-DXIL-DAG: %struct.S = type { <3 x float>, <4 x float> }
// CHECK-SPIR-DAG: %S = type <{ <3 x float>, target("spirv.Padding", 4), <4 x float> }>
// CHECK-SPIR-DAG: %struct.S = type { <3 x float>, <4 x float> }

cbuffer CB {
  S cbs;
};
// CHECK-DXIL-DAG: @cbs = external hidden addrspace(2) global %S, align 1
// CHECK-SPIR-DAG: @cbs = external hidden addrspace(12) global %S, align 1

void main() {
  S tmp = (S)cbs;
// CHECK-DXIL: %agg-temp = alloca %struct.S, align 1
// CHECK-DXIL: %[[#DST:]] = getelementptr inbounds %struct.S, ptr %agg-temp, i32 0, i32 0
// CHECK-DXIL: %cbuf.load = load <3 x float>, ptr addrspace(2) @cbs, align 4
// CHECK-DXIL: store <3 x float> %cbuf.load, ptr %[[#DST]], align 4

// CHECK-DXIL: %[[#DST:]] = getelementptr inbounds %struct.S, ptr %agg-temp, i32 0, i32 1
// CHECK-DXIL: %cbuf.load1 = load <4 x float>, ptr addrspace(2) getelementptr inbounds nuw (i8, ptr addrspace(2) @cbs, i32 16), align 4
// CHECK-DXIL: store <4 x float> %cbuf.load1, ptr %[[#DST]], align 4

// CHECK-SPIR: %agg-temp = alloca %struct.S, align 1
// CHECK-SPIR: %[[#DST:]] = getelementptr inbounds %struct.S, ptr %agg-temp, i32 0, i32 0
// CHECK-SPIR: %cbuf.load = load <3 x float>, ptr addrspace(12) @cbs, align 4
// CHECK-SPIR: store <3 x float> %cbuf.load, ptr %[[#DST]], align 4

// CHECK-SPIR: %[[#DST:]] = getelementptr inbounds %struct.S, ptr %agg-temp, i32 0, i32 1
// CHECK-SPIR: %cbuf.load1 = load <4 x float>, ptr addrspace(12) getelementptr inbounds nuw (i8, ptr addrspace(12) @cbs, i64 16), align 4
// CHECK-SPIR: store <4 x float> %cbuf.load1, ptr %[[#DST]], align 4
}
