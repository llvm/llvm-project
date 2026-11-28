// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry main -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// Without native 16-bit types, `half` is lowered to a 32-bit float, so it is a
// valid type for the PointSize built-in.

struct VSOutput {
  float4 pos : SV_Position;
  [[vk::builtin("PointSize")]] half size : PSIZE;
};

[shader("vertex")]
VSOutput main() {
  VSOutput output;
  output.pos = float4(0, 0, 0, 1);
  output.size = 1.0;
  return output;
}

// CHECK: @PSIZE = {{.*}}addrspace(8){{.*}}!spirv.Decorations ![[PSIZE_MD:[0-9]+]]
// CHECK: store float %{{.*}}, ptr addrspace(8) @PSIZE
// CHECK: ![[PSIZE_MD]] = !{![[PSIZE_DEC:[0-9]+]]}
// CHECK: ![[PSIZE_DEC]] = !{i32 11, i32 1}
