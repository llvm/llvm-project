// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -o - %s -verify

// expected-no-error
[shader("compute")][numthreads(32,1,1)]
void compute(int GI : SV_GroupIndex) {}

// expected-error@+2 {{semantic 'SV_GroupIndex' is not supported in pixel shader inputs}}
[shader("pixel")]
void pixel(int GI : SV_GroupIndex) {}

// expected-error@+2 {{semantic 'SV_GroupIndex' is not supported in vertex shader inputs}}
[shader("vertex")]
void vertex(int GI : SV_GroupIndex) {}

// expected-error@+2 {{semantic 'SV_GroupIndex' is not supported in geometry shader inputs}}
[shader("geometry")]
void geometry(int GI : SV_GroupIndex) {}

// expected-error@+2 {{semantic 'SV_GroupIndex' is not supported in domain shader inputs}}
[shader("domain")]
void domain(int GI : SV_GroupIndex) {}

[shader("amplification")][numthreads(32,1,1)]
void amplification(int GI : SV_GroupIndex) {}

[shader("mesh")][numthreads(32,1,1)]
void mesh(int GI : SV_GroupIndex) {}
