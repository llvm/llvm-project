// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex  -hlsl-entry vs_ok  -fsyntax-only -verify -DCASE_VS_OUT_OK %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex  -hlsl-entry vs_err -fsyntax-only -verify -DCASE_VS_IN_ERR %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel   -hlsl-entry ps_in  -fsyntax-only -verify -DCASE_PS_IN     %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel   -hlsl-entry ps_out -fsyntax-only -verify -DCASE_PS_OUT    %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -hlsl-entry cs     -fsyntax-only -verify -DCASE_CS        %s

// PointSize is legal as a vertex shader output; vertex shader input, pixel
// shader input and output, and compute are not.

#ifdef CASE_VS_OUT_OK
// expected-no-diagnostics
// Vertex output is the only legal vertex position.
struct Out {
  float4 pos : SV_Position;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};
[shader("vertex")] Out vs_ok() { Out o; o.pos = 0; o.size = 1; return o; }
#endif

#ifdef CASE_VS_IN_ERR
// expected-error@+1 {{'PointSize' builtin cannot be used as vertex shader inputs}}
float4 vs_err([[vk::builtin("PointSize")]] float size : PSIZE) : SV_Position {
  return size;
}
#endif

#ifdef CASE_PS_IN
// Vulkan does not allow BuiltIn PointSize in a fragment shader
// (VUID-PointSize-PointSize-04314).
// expected-error@+1 {{'PointSize' builtin cannot be used as pixel shader inputs}}
float4 ps_in([[vk::builtin("PointSize")]] float size : PSIZE) : SV_Target {
  return size;
}
#endif

#ifdef CASE_PS_OUT
struct Out {
  float4 c : SV_Target;
  // PSIZE is also not a valid pixel shader output semantic.
  // expected-error@+2 {{semantic 'PSIZE' is not supported in pixel shader outputs}}
  // expected-error@+1 {{'PointSize' builtin cannot be used as pixel shader outputs}}
  [[vk::builtin("PointSize")]] float size : PSIZE;
};
[shader("pixel")] Out ps_out() { Out o; o.c = 0; o.size = 1; return o; }
#endif

#ifdef CASE_CS
[numthreads(1, 1, 1)]
// expected-error@+2 {{semantic 'PSIZE' is not supported in compute shader inputs}}
// expected-error@+1 {{'PointSize' builtin cannot be used as compute shader inputs}}
void cs([[vk::builtin("PointSize")]] float size : PSIZE) {}
#endif
