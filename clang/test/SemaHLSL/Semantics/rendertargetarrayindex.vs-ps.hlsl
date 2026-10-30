// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Type, index and interpolation checks for SV_RenderTargetArrayIndex on vertex
// outputs and pixel inputs. These are independent of the shader stage and IO
// direction, which are covered by rendertargetarrayindex.stages.hlsl.

struct VSOut {
  float4 p : SV_Position;
  uint vp : SV_RenderTargetArrayIndex;
};
[shader("vertex")]
VSOut vertex_output() { return (VSOut)0; }
[shader("vertex")]
int vertex_return() : sv_rendertargetarrayindex0 { return 0; }
[shader("pixel")]
void pixel_input(uint vp : SV_RenderTargetArrayIndex) {}

// 16-bit integers are allowed. SPIR-V converts them to and from the 32-bit
// built-in.
[shader("pixel")]
void uint16(uint16_t vp : SV_RenderTargetArrayIndex) {}

[shader("pixel")]
void bad_float(float vp : SV_RenderTargetArrayIndex) {}
// expected-error@-1 {{semantic 'SV_RenderTargetArrayIndex' must be a scalar of 16 or 32 bit integer type (was 'float')}}
[shader("vertex")]
uint2 bad_vector() : SV_RenderTargetArrayIndex { return 0; }
// expected-error@-1 {{semantic 'SV_RenderTargetArrayIndex' must be a scalar of 16 or 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}
[shader("pixel")]
void bad_bool(bool vp : SV_RenderTargetArrayIndex) {}
// expected-error@-1 {{semantic 'SV_RenderTargetArrayIndex' must be a scalar of 16 or 32 bit integer type (was 'bool')}}

[shader("pixel")]
void bad_index(uint vp : SV_RenderTargetArrayIndex1) {}
// expected-error@-1 {{semantic 'SV_RenderTargetArrayIndex' does not allow indexing}}

// Integer inputs can only use nointerpolation.
[shader("pixel")]
void bad_interpolation(linear uint vp : SV_RenderTargetArrayIndex) {}
// expected-error@-1 {{interpolation modifiers other than 'nointerpolation' cannot be used with type 'uint' (aka 'unsigned int')}}
