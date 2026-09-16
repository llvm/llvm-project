// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

// SV_Target indices select one of eight render target slots.
[shader("pixel")]
float4 target_zero(float4 P : SV_Position) : SV_Target { return P; }

[shader("pixel")]
float4 target_last(float4 P : SV_Position) : SV_Target7 { return P; }

[shader("pixel")]
float4 target_past_last(float4 P : SV_Position) : SV_Target8 { return P; }
// expected-error@-1 {{semantic 'SV_Target' index 8 exceeds the maximum supported index 7}}

[shader("pixel")]
float4 target_way_past_last(float4 P : SV_Position) : SV_Target100 { return P; }
// expected-error@-1 {{semantic 'SV_Target' index 100 exceeds the maximum supported index 7}}

// Both array elements must fit in the render-target index range.
[shader("pixel")]
void target_array_fits(out float4 T[2] : SV_Target6) { T[0] = 0; T[1] = 0; }

[shader("pixel")]
void target_array_overflows(out float4 T[2] : SV_Target7) { T[0] = 0; T[1] = 0; }
// expected-error@-1 {{semantic 'SV_Target' index 8 exceeds the maximum supported index 7}}

// The return semantic assigns consecutive indices to the fields.
struct TwoTargets {
  float4 A;
  float4 B;
};

[shader("pixel")]
TwoTargets target_struct_fits() : SV_Target6 { return (TwoTargets)0; }

[shader("pixel")]
TwoTargets target_struct_overflows() : SV_Target7 { return (TwoTargets)0; }
// expected-error@-1 {{semantic 'SV_Target' index 8 exceeds the maximum supported index 7}}
