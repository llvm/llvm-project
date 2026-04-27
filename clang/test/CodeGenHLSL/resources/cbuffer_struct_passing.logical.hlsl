// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s -fexperimental-logical-pointer | FileCheck %s --check-prefixes=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s -fexperimental-logical-pointer | FileCheck %s --check-prefixes=CHECK-SPIR

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
// CHECK-DXIL: %agg-temp = call elementtype(%struct.S) ptr @llvm.structured.alloca.p0()
// CHECK-DXIL: %[[#SRC:]] = call ptr addrspace(2) (ptr addrspace(2), ...) @llvm.structured.gep.p2(ptr addrspace(2) elementtype(%S) @cbs, i32 0)
// CHECK-DXIL: %[[#DST:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %agg-temp, i32 0)
// CHECK-DXIL: %cbuf.load = load <3 x float>, ptr addrspace(2) %[[#SRC]], align 4
// CHECK-DXIL: store <3 x float> %cbuf.load, ptr %[[#DST]], align 4

// CHECK-DXIL: %[[#SRC:]] = call ptr addrspace(2) (ptr addrspace(2), ...) @llvm.structured.gep.p2(ptr addrspace(2) elementtype(%S) @cbs, i32 2)
// CHECK-DXIL: %[[#DST:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %agg-temp, i32 1)
// CHECK-DXIL: %cbuf.load1 = load <4 x float>, ptr addrspace(2) %[[#SRC]], align 4
// CHECK-DXIL: store <4 x float> %cbuf.load1, ptr %[[#DST]], align 4

// CHECK-SPIR: %agg-temp = call elementtype(%struct.S) ptr @llvm.structured.alloca.p0()
// CHECK-SPIR: %[[#SRC:]] = call ptr addrspace(12) (ptr addrspace(12), ...) @llvm.structured.gep.p12(ptr addrspace(12) elementtype(%S) @cbs, i32 0)
// CHECK-SPIR: %[[#DST:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %agg-temp, i32 0)
// CHECK-SPIR: %cbuf.load = load <3 x float>, ptr addrspace(12) %[[#SRC]], align 4
// CHECK-SPIR: store <3 x float> %cbuf.load, ptr %[[#DST]], align 4

// CHECK-SPIR: %[[#SRC:]] = call ptr addrspace(12) (ptr addrspace(12), ...) @llvm.structured.gep.p12(ptr addrspace(12) elementtype(%S) @cbs, i32 2)
// CHECK-SPIR: %[[#DST:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %agg-temp, i32 1)
// CHECK-SPIR: %cbuf.load1 = load <4 x float>, ptr addrspace(12) %[[#SRC]], align 4
// CHECK-SPIR: store <4 x float> %cbuf.load1, ptr %[[#DST]], align 4
}
