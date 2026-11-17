// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// A pixel shader PointSize input is a BuiltIn PointSize input and leaves the
// other locations unchanged.

struct VSOutput {
  float4 pos : SV_Position;
  [[vk::location(0)]] float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
  [[vk::location(1)]] float b : B;
};

[shader("pixel")]
float4 main(VSOutput input) : SV_Target {
  return float4(input.a, input.b, 0, 1);
}

// CHECK: @A0 = {{.*}}!spirv.Decorations ![[A_MD:[0-9]+]]
// CHECK: @PSIZE = {{.*}}addrspace(7){{.*}}!spirv.Decorations ![[PSIZE_MD:[0-9]+]]
// CHECK: @B0 = {{.*}}!spirv.Decorations ![[B_MD:[0-9]+]]
// CHECK: define void @main()
// CHECK: load float, ptr addrspace(7) @PSIZE

// CHECK: ![[A_MD]] = !{![[A_DEC:[0-9]+]]}
// CHECK: ![[A_DEC]] = !{i32 30, i32 0}
// CHECK: ![[PSIZE_MD]] = !{![[PSIZE_DEC:[0-9]+]]}
// CHECK: ![[PSIZE_DEC]] = !{i32 11, i32 1}
// CHECK: ![[B_MD]] = !{![[B_DEC:[0-9]+]]}
// CHECK: ![[B_DEC]] = !{i32 30, i32 1}
