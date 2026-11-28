// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Type and index checks for SV_StencilRef. These are independent of the shader
// stage and IO direction, which are covered by stencilref.stages.hlsl.

[shader("pixel")]
uint valid_uint() : SV_StencilRef { return 0; }
[shader("pixel")]
int valid_int() : sv_stencilref0 { return 0; }

struct Output {
  float4 c : SV_Target;
  uint s : SV_StencilRef;
};
[shader("pixel")]
Output valid_struct() { return (Output)0; }

// 16-bit integers are allowed. SPIR-V converts them to and from the 32-bit
// built-in.
[shader("pixel")]
uint16_t uint16() : SV_StencilRef { return 0; }

[shader("pixel")]
float bad_float() : SV_StencilRef { return 0; }
// expected-error@-1 {{semantic 'SV_StencilRef' must be a scalar of 16 or 32 bit integer type (was 'float')}}
[shader("pixel")]
bool bad_bool() : SV_StencilRef { return 0; }
// expected-error@-1 {{semantic 'SV_StencilRef' must be a scalar of 16 or 32 bit integer type (was 'bool')}}
[shader("pixel")]
uint64_t bad_int64() : SV_StencilRef { return 0; }
// expected-error@-1 {{semantic 'SV_StencilRef' must be a scalar of 16 or 32 bit integer type (was 'uint64_t' (aka 'unsigned long'))}}
[shader("pixel")]
uint2 bad_vector() : SV_StencilRef { return 0; }
// expected-error@-1 {{semantic 'SV_StencilRef' must be a scalar of 16 or 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}
struct ArrayOutput {
  uint s[1] : SV_StencilRef;
  // expected-error@-1 {{semantic 'SV_StencilRef' must be a scalar of 16 or 32 bit integer type (was 'uint[1]'}}
};
[shader("pixel")]
ArrayOutput bad_array() { return (ArrayOutput)0; }

[shader("pixel")]
uint bad_index() : SV_StencilRef1 { return 0; }
// expected-error@-1 {{semantic 'SV_StencilRef' does not allow indexing}}

struct TwoStencils {
  uint s : SV_StencilRef; // expected-note {{previous use is here}}
  uint t : SV_StencilRef0; // expected-note {{'t' used here}}
  // expected-error@-1 {{semantic index overlap SV_StencilRef0}}
};
[shader("pixel")]
TwoStencils same_index() { return (TwoStencils)0; }
