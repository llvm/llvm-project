// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

struct Pair {
  uint A;
  uint B;
};

// The semantic index can be written explicitly ...
[shader("compute")][numthreads(1,1,1)]
void explicit_index(uint GI : SV_GroupIndex1) {}
// expected-error@-1 {{semantic 'SV_GroupIndex' does not allow indexing}}

// ... or be derived when a semantic is spread over an aggregate.
[shader("compute")][numthreads(1,1,1)]
void derived_index(Pair GI : SV_GroupIndex) {}
// expected-error@-1 {{semantic 'SV_GroupIndex' does not allow indexing}}

[shader("compute")][numthreads(1,1,1)]
void no_index(uint GI : SV_GroupIndex) {}

// An array also derives an index per element.
[shader("compute")][numthreads(1,1,1)]
void array_index(uint3 ID[2] : SV_DispatchThreadID) {}
// expected-error@-1 {{semantic 'SV_DispatchThreadID' does not allow indexing}}

// SV_Position is non-indexable on pixel shader inputs.
[shader("pixel")]
float4 position_ps(float4 P : SV_Position1) : SV_Target { return P; }
// expected-error@-1 {{semantic 'SV_Position' does not allow indexing}}

// On vertex shader inputs, SV_Position is arbitrary and may be indexed.
[shader("vertex")]
float4 position_vs(float4 P : SV_Position1) : SV_Position { return P; }

[shader("pixel")]
float4 user_index(float4 P : USER7) : SV_Target { return P; }

// Clip/cull indices are allowed even with a system-value interpretation.
[shader("pixel")]
float4 clip_cull_index(float Clip : SV_ClipDistance1,
                      float Cull : sv_culldistance1) : SV_Target {
  return Clip + Cull;
}

// Recognizing the name does not bypass shader-stage validation.
[shader("compute")][numthreads(1,1,1)]
void clip_compute(float Clip : SV_ClipDistance1) {}
// expected-error@-1 {{semantic 'SV_ClipDistance' is not supported in compute shader inputs}}
