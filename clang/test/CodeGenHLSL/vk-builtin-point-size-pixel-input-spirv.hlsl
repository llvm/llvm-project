// REQUIRES: spirv-registered-target

// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -S -o - %s | FileCheck %s --check-prefix=UNREAD --implicit-check-not=PointSize
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -DDEAD -S -o - %s | FileCheck %s --check-prefix=UNREAD --implicit-check-not=PointSize
// RUN: not %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O0 -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -DLIVE -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -DHELPER -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -DSCALAR -S -o - %s | FileCheck %s --check-prefix=UNREAD --implicit-check-not=PointSize
// RUN: not %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -DSCALAR -DLIVE -S -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: %if spirv-val %{ %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel -hlsl-entry main -O3 -emit-obj -o %t.spv %s %}
// RUN: %if spirv-val %{ spirv-val --target-env vulkan1.3 %t.spv %}

// A pixel shader PointSize input compiles when optimization removes it, and is
// an error when the shader uses it or at -O0. It does not take a location.

// UNREAD-DAG: OpEntryPoint Fragment
// UNREAD-DAG: OpDecorate %[[#]] Location 0
// UNREAD-DAG: OpDecorate %[[#]] Location 1

// ERROR: error: BuiltIn PointSize cannot be used in a Fragment entry point (VUID-PointSize-PointSize-04314)

#ifdef SCALAR
[shader("pixel")]
float4 main(float a : A, [[vk::builtin("PointSize")]] float size : PSIZE,
            float b : B) : SV_Target {
#ifdef LIVE
  return size;
#else
  return float4(a, b, 0, 1);
#endif
}
#else
struct Shared {
  float a : A;
  [[vk::builtin("PointSize")]] float size : PSIZE;
  float b : B;
};

float readSize(Shared input) { return input.size; }

[shader("pixel")]
float4 main(Shared input) : SV_Target {
#if defined(LIVE)
  return input.size;
#elif defined(HELPER)
  return readSize(input);
#else
#ifdef DEAD
  float dead = input.size;
#endif
  return float4(input.a, input.b, 0, 1);
#endif
}
#endif
