// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-library -verify %s

// expected-no-diagnostics
[shader("compute")]
[numthreads(8,8,1)]
unsigned foo() {
    return hlsl::WaveGetLaneCount();
}

