// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -o - %s -verify

// expected-no-diagnostics

[shader("pixel")] void pixel() {}
[shader("vertex")] void vertex() {}
[shader("raygeneration")] void raygeneration() {}
[shader("intersection")] void intersection() {}

[numthreads(1,1,1)][shader("compute")] void compute() {}
[numthreads(1,1,1)][shader("mesh")] void mesh() {}

// Note: the rest of these have additional constraints that aren't implemented
// yet, so here we just declare them to make sure the spelling works and
// whatnot.
[shader("geometry")] void geometry();
[shader("hull")] void hull();
[shader("domain")] void domain();
[shader("callable")] void callable();
[shader("closesthit")] void closesthit();
[shader("anyhit")] void anyhit();
[shader("miss")] void miss();

[numthreads(1,1,1)][shader("amplification")] void amplification();
