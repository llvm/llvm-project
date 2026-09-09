// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

// Parameters passed by reference are written by the entry point, so they are
// verified against the output signature of the shader stage.

[shader("vertex")]
void vs_out(out float4 Pos : SV_Position) { Pos = 0; }

[shader("pixel")]
void ps_out(out float4 Color : SV_Target) { Color = 0; }

[shader("pixel")]
void ps_position_out(out float4 Pos : SV_Position) { Pos = 0; }
// expected-error@-1 {{semantic 'SV_Position' is not supported as pixel shader outputs; it is only available as an input}}

[shader("compute")][numthreads(1,1,1)]
void cs_group_index_out(out uint GI : SV_GroupIndex) { GI = 0; }
// expected-error@-1 {{semantic 'SV_GroupIndex' is not supported as compute shader outputs; it is only available as an input}}

// Output parameters share the output signature with the return value.
[shader("pixel")]
float4 ps_overlap(out float4 Color : SV_Target) : SV_Target {
// expected-error@-1 {{semantic index overlap SV_Target0}}
  Color = 0;
  return 0;
}
