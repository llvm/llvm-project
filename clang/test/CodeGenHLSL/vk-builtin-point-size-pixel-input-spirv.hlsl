// REQUIRES: spirv-registered-target

// RUN: split-file %s %t
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o - %t/unread.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o - %t/dead.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o - %t/scalar.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O0 -S -o - %t/scalar.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-library -O3 -S -o - %t/library.hlsl | FileCheck %s --check-prefix=LIB --implicit-check-not="BuiltIn PointSize"

// RUN: %if spirv-val %{ %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -emit-obj -o %t/unread.spv %t/unread.hlsl %}
// RUN: %if spirv-val %{ spirv-val --target-env vulkan1.3 %t/unread.spv %}
// RUN: %if spirv-val %{ %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O0 -emit-obj -o %t/scalar.spv %t/scalar.hlsl %}
// RUN: %if spirv-val %{ spirv-val --target-env vulkan1.3 %t/scalar.spv %}
// RUN: %if spirv-val %{ %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-library -O3 -emit-obj -o %t/library.spv %t/library.hlsl %}
// RUN: %if spirv-val %{ spirv-val --target-env vulkan1.3 %t/library.spv %}

// A pixel shader PointSize input that optimization removes compiles. The
// checks establish that PointSize is gone and that the input does not take a
// location: B keeps Location 1 and nothing takes Location 2. In a module with
// both stages, only the vertex shader's PointSize output remains. The error
// cases are in vk-builtin-point-size-pixel-input-spirv-errors.hlsl.

// CHECK-DAG: OpEntryPoint Fragment
// CHECK-DAG: OpDecorate %[[#]] Location 0
// CHECK-DAG: OpDecorate %[[#]] Location 1

// LIB-DAG: OpEntryPoint Vertex %[[#]] "vs_main"
// LIB-DAG: OpEntryPoint Fragment %[[#]] "ps_main"
// LIB-DAG: OpDecorate %[[#]] BuiltIn PointSize

//--- unread.hlsl
struct Shared {
  float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
  float b : B;
};

[shader("pixel")]
float4 main(Shared input) : SV_Target {
  return float4(input.a, input.b, 0, 1);
}

//--- dead.hlsl
struct Shared {
  float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
  float b : B;
};

[shader("pixel")]
float4 main(Shared input) : SV_Target {
  float dead = input.size;
  return float4(input.a, input.b, 0, 1);
}

//--- scalar.hlsl
// Unlike an unread struct field, an unread scalar also compiles at -O0.
[shader("pixel")]
float4 main(float a : A, [[vk::builtin("PointSize")]] float size : PSIZE,
            float b : B) : SV_Target {
  return float4(a, b, 0, 1);
}

//--- library.hlsl
struct Shared {
  float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

[shader("vertex")]
Shared vs_main(float a : A) {
  Shared output;
  output.a = a;
  output.size = 1.0;
  return output;
}

[shader("pixel")]
float4 ps_main(Shared input) : SV_Target {
  return float4(input.a, 0, 0, 1);
}
