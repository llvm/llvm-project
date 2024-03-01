// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -verify %s

// expected-error@+1 {{expected string literal as argument of 'RootSignature' attribute}}
[RootSignature(1)]
[shader("compute")]
[numthreads(1,1,1)]
void main() {}
