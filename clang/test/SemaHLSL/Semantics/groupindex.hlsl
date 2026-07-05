// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -o - %s -verify

// expected-no-error
[shader("compute")][numthreads(32,1,1)]
void compute(int GI : SV_GroupIndex) {}

// expected-error@+2 {{attribute 'SV_GroupIndex' is unsupported in 'pixel' shaders}}
[shader("pixel")]
void pixel(int GI : SV_GroupIndex) {}

// expected-error@+2 {{attribute 'SV_GroupIndex' is unsupported in 'vertex' shaders}}
[shader("vertex")]
void vertex(int GI : SV_GroupIndex) {}

// expected-error@+2 {{attribute 'SV_GroupIndex' is unsupported in 'geometry' shaders}}
[shader("geometry")]
void geometry(int GI : SV_GroupIndex) {}

// expected-error@+2 {{attribute 'SV_GroupIndex' is unsupported in 'domain' shaders}}
[shader("domain")]
void domain(int GI : SV_GroupIndex) {}

// expected-error@+2 {{attribute 'SV_GroupIndex' is unsupported in 'amplification' shaders}}
[shader("amplification")][numthreads(32,1,1)]
void amplification(int GI : SV_GroupIndex) {}

// expected-error@+2 {{attribute 'SV_GroupIndex' is unsupported in 'mesh' shaders}}
[shader("mesh")][numthreads(32,1,1)]
void mesh(int GI : SV_GroupIndex) {}
