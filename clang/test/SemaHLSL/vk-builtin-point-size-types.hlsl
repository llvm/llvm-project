// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_float  -fsyntax-only -verify -DCASE_FLOAT  %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_half   -fsyntax-only -verify -DCASE_HALF   %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_double -fsyntax-only -verify -DCASE_DOUBLE %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_vector -fsyntax-only -verify -DCASE_VECTOR %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_int    -fsyntax-only -verify -DCASE_INT    %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_bool   -fsyntax-only -verify -DCASE_BOOL   %s

// PointSize accepts any scalar floating-point type, and rejects everything
// else (vectors, integers, bool).

#ifdef CASE_FLOAT
// expected-no-diagnostics
struct Out {
  float4 pos : SV_Position;
  [[vk::builtin("PointSize")]] float size : PSIZE;
};
[shader("vertex")] Out vs_float() { Out o; o.pos = 0; o.size = 1; return o; }
#endif

#ifdef CASE_HALF
// expected-no-diagnostics
struct Out {
  float4 pos : SV_Position;
  [[vk::builtin("PointSize")]] half size : PSIZE;
};
[shader("vertex")] Out vs_half() { Out o; o.pos = 0; o.size = 1; return o; }
#endif

#ifdef CASE_DOUBLE
// expected-no-diagnostics
struct Out {
  float4 pos : SV_Position;
  [[vk::builtin("PointSize")]] double size : PSIZE;
};
[shader("vertex")] Out vs_double() { Out o; o.pos = 0; o.size = 1; return o; }
#endif

#ifdef CASE_VECTOR
struct Out {
  float4 pos : SV_Position;
  // expected-error@+1 {{'PointSize' builtin must be of type 'float'}}
  [[vk::builtin("PointSize")]] float2 size : PSIZE;
};
[shader("vertex")] Out vs_vector() { Out o; o.pos = 0; o.size = 1; return o; }
#endif

#ifdef CASE_INT
struct Out {
  float4 pos : SV_Position;
  // expected-error@+1 {{'PointSize' builtin must be of type 'float'}}
  [[vk::builtin("PointSize")]] int size : PSIZE;
};
[shader("vertex")] Out vs_int() { Out o; o.pos = 0; o.size = 1; return o; }
#endif

#ifdef CASE_BOOL
struct Out {
  float4 pos : SV_Position;
  // expected-error@+1 {{'PointSize' builtin must be of type 'float'}}
  [[vk::builtin("PointSize")]] bool size : PSIZE;
};
[shader("vertex")] Out vs_bool() { Out o; o.pos = 0; o.size = true; return o; }
#endif
