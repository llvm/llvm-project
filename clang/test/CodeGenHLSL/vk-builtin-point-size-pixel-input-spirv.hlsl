// REQUIRES: spirv-registered-target

// RUN: split-file %s %t
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o - %t/unread.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o - %t/dead.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o - %t/scalar.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O0 -S -o - %t/scalar.hlsl | FileCheck %s --implicit-check-not=PointSize --implicit-check-not="Location 2"

// RUN: %if spirv-val %{ %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -emit-obj -o %t/unread.spv %t/unread.hlsl %}
// RUN: %if spirv-val %{ spirv-val --target-env vulkan1.3 %t/unread.spv %}
// RUN: %if spirv-val %{ %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O0 -emit-obj -o %t/scalar.spv %t/scalar.hlsl %}
// RUN: %if spirv-val %{ spirv-val --target-env vulkan1.3 %t/scalar.spv %}

// A pixel shader PointSize input that optimization removes compiles. The
// checks establish that PointSize is gone and that the input does not take a
// location: B keeps Location 1 and nothing takes Location 2. The error cases
// are in vk-builtin-point-size-pixel-input-spirv-errors.hlsl.

// CHECK-DAG: OpEntryPoint Fragment
// CHECK-DAG: OpDecorate %[[#]] Location 0
// CHECK-DAG: OpDecorate %[[#]] Location 1

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
