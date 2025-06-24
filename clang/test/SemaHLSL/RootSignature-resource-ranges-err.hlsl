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
