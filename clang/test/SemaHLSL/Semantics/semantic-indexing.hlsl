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
