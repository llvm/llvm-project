// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

// expected-error@+1 {{resource ranges b[0;2] and b[2;2] overlap within space = 0 and visibility = All}}
[numthreads(8, 8, 1)]
[RootSignature("RootConstants(num32BitConstants=4, b2), DescriptorTable(CBV(b10, numDescriptors=3))")]
void bad_root_signature_9() {}
