// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Invalid type, index and interpolation uses of SV_ViewID. The valid cases are
// covered by viewid.ps.hlsl, and the shader stage and IO direction by
// viewid.stages.hlsl.

[shader("pixel")]
void bad_float(float v : SV_ViewID) {}
// expected-error@-1 {{semantic 'SV_ViewID' must be a scalar of 32 bit integer type (was 'float')}}
[shader("pixel")]
void bad_bool(bool v : SV_ViewID) {}
// expected-error@-1 {{semantic 'SV_ViewID' must be a scalar of 32 bit integer type (was 'bool')}}
[shader("pixel")]
void bad_uint16(uint16_t v : SV_ViewID) {}
// expected-error@-1 {{semantic 'SV_ViewID' must be a scalar of 32 bit integer type (was 'uint16_t' (aka 'unsigned short'))}}
[shader("pixel")]
void bad_vector(uint2 v : SV_ViewID) {}
// expected-error@-1 {{semantic 'SV_ViewID' must be a scalar of 32 bit integer type (was 'uint2' (aka 'vector<uint, 2>'))}}
[shader("pixel")]
void bad_array(uint v[1] : SV_ViewID) {}
// expected-error@-1 {{semantic 'SV_ViewID' must be a scalar of 32 bit integer type (was 'uint[1]'}}

[shader("pixel")]
void bad_index(uint v : SV_ViewID1) {}
// expected-error@-1 {{semantic 'SV_ViewID' does not allow indexing}}

[shader("pixel")]
void same_index(uint v : SV_ViewID, // expected-note {{previous use is here}}
                uint w : SV_ViewID0) {} // expected-note {{'w' declared here}}
// expected-error@-1 {{semantic index overlap SV_ViewID0}}

// Integer inputs can only use nointerpolation.
[shader("pixel")]
void bad_interpolation(linear uint v : SV_ViewID) {}
// expected-error@-1 {{interpolation modifiers other than 'nointerpolation' cannot be used with type 'uint' (aka 'unsigned int')}}
