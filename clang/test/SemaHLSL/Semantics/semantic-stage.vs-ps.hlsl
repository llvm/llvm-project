// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

[shader("vertex")]
uint vs_vertexid_out(uint ID : SV_VertexID) : SV_VertexID { return ID; }
// expected-error@-1 {{semantic 'SV_VertexID' is not supported in vertex shader outputs}}

[shader("pixel")]
float4 ps_dispatchthreadid_in(uint3 ID : SV_DispatchThreadID) : SV_Target { return 0; }
// expected-error@-1 {{semantic 'SV_DispatchThreadID' is not supported in pixel shader inputs}}

[shader("pixel")]
float4 ps_groupindex_in(uint GI : SV_GroupIndex) : SV_Target { return 0; }
// expected-error@-1 {{semantic 'SV_GroupIndex' is not supported in pixel shader inputs}}

[shader("pixel")]
void ps_vertexid_out(out uint ID : SV_VertexID) { ID = 0; }
// expected-error@-1 {{semantic 'SV_VertexID' is not supported in pixel shader outputs}}

// The same field is arbitrary in a vertex input and a system value in a pixel
// input. Validate it at the entry point rather than when declaring the field.
struct IntegerPosition {
  int4 P : SV_Position;
  // expected-error@-1 {{semantic 'SV_Position' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type}}
};

[shader("vertex")]
float4 vs_position_in(IntegerPosition I) : SV_Position { return (float4)I.P; }

[shader("pixel")]
float4 ps_position_in(IntegerPosition I) : SV_Target { return (float4)I.P; }

// Vertex outputs are system values, so integer components remain invalid.
[shader("vertex")]
int4 vs_position_out(float4 P : USER) : SV_Position { return (int4)P; }
// expected-error@-1 {{semantic 'SV_Position' must be a scalar or vector of up to 4 components of 16 or 32 bit floating-point type}}

// Custom semantics allow explicit indices on inputs, out parameters, and
// return values; none should enter the system-only index checker.
[shader("vertex")]
float4 vs_user(float4 A : MYSEMANTIC5, out float4 B : OUT1) : OUT2 {
  B = A;
  return A;
}

[shader("pixel")]
float4 ps_user_in(float4 A : IN) : SV_Target { return A; }
