// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

#define Overlap0 "CBV(b42), CBV(b42)"

[RootSignature(Overlap0)] // expected-error {{resource ranges b[42;42] and b[42;42] overlap within space = 0 and visibility = All}}
void bad_root_signature_0() {}

#define Overlap1 "SRV(t0, space = 3), SRV(t0, space = 3)"

[RootSignature(Overlap1)] // expected-error {{resource ranges t[0;0] and t[0;0] overlap within space = 3 and visibility = All}}
void bad_root_signature_1() {}

#define Overlap2 "UAV(u0, visibility = SHADER_VISIBILITY_PIXEL), UAV(u0, visibility = SHADER_VISIBILITY_PIXEL)"

[RootSignature(Overlap2)] // expected-error {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
void bad_root_signature_2() {}

#define Overlap3 "UAV(u0, visibility = SHADER_VISIBILITY_ALL), UAV(u0, visibility = SHADER_VISIBILITY_PIXEL)"

[RootSignature(Overlap3)] // expected-error {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
void bad_root_signature_3() {}

#define Overlap4 "UAV(u0, visibility = SHADER_VISIBILITY_PIXEL), UAV(u0, visibility = SHADER_VISIBILITY_ALL)"

[RootSignature(Overlap4)] // expected-error {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
void bad_root_signature_4() {}
