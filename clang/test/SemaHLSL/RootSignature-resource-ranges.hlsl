// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

// expected-no-diagnostics

[RootSignature("CBV(b0), CBV(b1)")]
void valid_root_signature_0() {}

[RootSignature("CBV(b0, visibility = SHADER_VISIBILITY_DOMAIN), CBV(b0, visibility = SHADER_VISIBILITY_PIXEL)")]
void valid_root_signature_1() {}

[RootSignature("CBV(b0, space = 1), CBV(b0, space = 2)")]
void valid_root_signature_2() {}

[RootSignature("CBV(b0), SRV(t0)")]
void valid_root_signature_3() {}

[RootSignature("RootConstants(num32BitConstants=4, b0, space=0), DescriptorTable(CBV(b0, space=1))")]
void valid_root_signature_4() {}

[RootSignature("StaticSampler(s2, visibility=SHADER_VISIBILITY_PIXEL), DescriptorTable(Sampler(s2), visibility=SHADER_VISIBILITY_VERTEX)")]
void valid_root_signature_5() {}

[RootSignature("DescriptorTable(SRV(t5), UAV(u5, numDescriptors=2))")]
void valid_root_signature_6() {}

[RootSignature("DescriptorTable(CBV(b0, offset = 4294967292), CBV(b1, numDescriptors = 3))")]
void valid_root_signature_7() {}
