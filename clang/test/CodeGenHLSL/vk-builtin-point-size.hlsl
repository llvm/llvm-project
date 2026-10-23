// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry main -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.8-vertex -hlsl-entry main -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=DXIL

// On SPIR-V, [[vk::builtin("PointSize")]] lowers to a BuiltIn PointSize
// decoration instead of an ordinary Location-based user semantic. The
// attribute doesn't exist on DXIL, so there PSIZE is an ordinary semantic.

struct VSOutput {
  float4 pos : SV_Position;
  float4 color : COLOR;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

[shader("vertex")]
VSOutput main() {
  VSOutput output = (VSOutput)0;
  return output;
}

// CHECK: @COLOR0 = {{.*}}!spirv.Decorations ![[COLOR_MD:[0-9]+]]
// CHECK: @PSIZE = {{.*}}addrspace(8){{.*}}!spirv.Decorations ![[PSIZE_MD:[0-9]+]]
// CHECK: define void @main()
// CHECK: store float %{{.*}}, ptr addrspace(8) @PSIZE

// BuiltIn decoration (11) with value 1 (PointSize), vs. COLOR0's Location (30).
// CHECK: ![[COLOR_MD]] = !{![[COLOR_DEC:[0-9]+]]}
// CHECK: ![[COLOR_DEC]] = !{i32 30,
// CHECK: ![[PSIZE_MD]] = !{![[PSIZE_DEC:[0-9]+]]}
// CHECK: ![[PSIZE_DEC]] = !{i32 11, i32 1}

// DXIL: define void @main()
// DXIL-NOT: PointSize
// DXIL: !{{[0-9]+}} = !{i32 {{[0-9]+}}, !"PSIZE",
