// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_float  -fsyntax-only -verify -DCASE_FLOAT  %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_half   -fsyntax-only -verify -DCASE_HALF   %s
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -fnative-int16-type -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_half -fsyntax-only -verify -DCASE_HALF_NATIVE %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_double -fsyntax-only -verify -DCASE_DOUBLE %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_vector -fsyntax-only -verify -DCASE_VECTOR %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_int    -fsyntax-only -verify -DCASE_INT    %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs_bool   -fsyntax-only -verify -DCASE_BOOL   %s

// Vulkan requires PointSize to be a 32-bit float (VUID-PointSize-PointSize-04317),
// so every other type is rejected: native half, double, vectors, integers and
// bool. Without native 16-bit types `half` is lowered to a 32-bit float, so it
// is accepted.

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

#ifdef CASE_HALF_NATIVE
struct Out {
  float4 pos : SV_Position;
  // expected-error@+1 {{'PointSize' builtin must be of type 'float'}}
  [[vk::builtin("PointSize")]] half size : PSIZE;
};
[shader("vertex")] Out vs_half() { Out o; o.pos = 0; o.size = 1; return o; }
#endif

#ifdef CASE_DOUBLE
struct Out {
  float4 pos : SV_Position;
  // expected-error@+1 {{'PointSize' builtin must be of type 'float'}}
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
