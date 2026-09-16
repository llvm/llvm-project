// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

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

// SV_Position is arbitrary on vertex inputs, so int2 is accepted.
[shader("vertex")]
float4 vs_position_in(int2 P : SV_Position) : SV_Position { return 0; }

[shader("vertex")]
float4 vs_user(float4 A : IN) : OUT { return A; }

[shader("pixel")]
float4 ps_user_in(float4 A : IN) : SV_Target { return A; }
