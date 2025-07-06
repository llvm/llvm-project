// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

// expected-error@+1 {{resource ranges b[42;42] and b[42;42] overlap within space = 0 and visibility = All}}
[RootSignature("CBV(b42), CBV(b42)")]
void bad_root_signature_0() {}

// expected-error@+1 {{resource ranges t[0;0] and t[0;0] overlap within space = 3 and visibility = All}}
[RootSignature("SRV(t0, space = 3), SRV(t0, space = 3)")]
void bad_root_signature_1() {}

// expected-error@+1 {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
[RootSignature("UAV(u0, visibility = SHADER_VISIBILITY_PIXEL), UAV(u0, visibility = SHADER_VISIBILITY_PIXEL)")]
void bad_root_signature_2() {}

// expected-error@+1 {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
[RootSignature("UAV(u0, visibility = SHADER_VISIBILITY_ALL), UAV(u0, visibility = SHADER_VISIBILITY_PIXEL)")]
void bad_root_signature_3() {}

// expected-error@+1 {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
[RootSignature("UAV(u0, visibility = SHADER_VISIBILITY_PIXEL), UAV(u0, visibility = SHADER_VISIBILITY_ALL)")]
void bad_root_signature_4() {}

// expected-error@+1 {{resource ranges b[0;0] and b[0;0] overlap within space = 0 and visibility = All}}
[RootSignature("RootConstants(num32BitConstants=4, b0), RootConstants(num32BitConstants=2, b0)")]
void bad_root_signature_5() {}

// expected-error@+1 {{resource ranges s[3;3] and s[3;3] overlap within space = 0 and visibility = All}}
[RootSignature("StaticSampler(s3), StaticSampler(s3)")]
void bad_root_signature_6() {}

// expected-error@+1 {{resource ranges t[2;5] and t[0;3] overlap within space = 0 and visibility = All}}
[RootSignature("DescriptorTable(SRV(t0, numDescriptors=4), SRV(t2, numDescriptors=4))")]
void bad_root_signature_7() {}

// expected-error@+1 {{resource ranges u[2;5] and u[0;unbounded) overlap within space = 0 and visibility = Hull}}
[RootSignature("DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility = SHADER_VISIBILITY_HULL), DescriptorTable(UAV(u2, numDescriptors=4))")]
void bad_root_signature_8() {}

// expected-error@+1 {{resource ranges b[0;2] and b[2;2] overlap within space = 0 and visibility = All}}
[RootSignature("RootConstants(num32BitConstants=4, b2), DescriptorTable(CBV(b0, numDescriptors=3))")]
void bad_root_signature_9() {}

// expected-error@+1 {{resource ranges s[4;unbounded) and s[17;17] overlap within space = 0 and visibility = All}}
[RootSignature("StaticSampler(s17), DescriptorTable(Sampler(s0, numDescriptors=3),Sampler(s4, numDescriptors=unbounded))")]
void bad_root_signature_10() {}

// expected-error@+1 {{resource ranges b[45;45] and b[4;unbounded) overlap within space = 0 and visibility = Geometry}}
[RootSignature("DescriptorTable(CBV(b4, numDescriptors=unbounded)), CBV(b45, visibility = SHADER_VISIBILITY_GEOMETRY)")]
void bad_root_signature_11() {}

#define ReportFirstOverlap \
 "DescriptorTable( " \
 "  CBV(b4, numDescriptors = 4), " \
 "  CBV(b1, numDescriptors = 2), " \
 "  CBV(b0, numDescriptors = 8), " \
 ")"

// expected-error@+1 {{resource ranges b[0;7] and b[1;2] overlap within space = 0 and visibility = All}}
[RootSignature(ReportFirstOverlap)]
void bad_root_signature_12() {}

// expected-error@+1 {{resource ranges s[2;2] and s[2;2] overlap within space = 0 and visibility = Vertex}}
[RootSignature("StaticSampler(s2, visibility=SHADER_VISIBILITY_ALL), DescriptorTable(Sampler(s2), visibility=SHADER_VISIBILITY_VERTEX)")]
void valid_root_signature_13() {}
