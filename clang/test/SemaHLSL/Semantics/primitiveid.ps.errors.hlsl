// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Invalid type, index and interpolation uses of SV_PrimitiveID. The valid
// cases are covered by primitiveid.ps.hlsl, and the shader stage and IO
// direction by primitiveid.stages.hlsl.

[shader("pixel")]
void bad_float(float s : SV_PrimitiveID) {}
// expected-error@-1 {{semantic 'SV_PrimitiveID' must be a scalar of 16 or 32 bit integer type (was 'float')}}
[shader("pixel")]
void bad_bool(bool s : SV_PrimitiveID) {}
// expected-error@-1 {{semantic 'SV_PrimitiveID' must be a scalar of 16 or 32 bit integer type (was 'bool')}}
[shader("pixel")]
void bad_int64(uint64_t s : SV_PrimitiveID) {}
// expected-error@-1 {{semantic 'SV_PrimitiveID' must be a scalar of 16 or 32 bit integer type (was 'uint64_t' (aka 'unsigned long'))}}
[shader("pixel")]
void bad_vector(uint1 s : SV_PrimitiveID) {}
// expected-error@-1 {{semantic 'SV_PrimitiveID' must be a scalar of 16 or 32 bit integer type (was 'uint1' (aka 'vector<uint, 1>'))}}
[shader("pixel")]
void bad_array(uint s[1] : SV_PrimitiveID) {}
// expected-error@-1 {{semantic 'SV_PrimitiveID' must be a scalar of 16 or 32 bit integer type (was 'uint[1]'}}

[shader("pixel")]
void bad_index(uint s : SV_PrimitiveID1) {}
// expected-error@-1 {{semantic 'SV_PrimitiveID' does not allow indexing}}

[shader("pixel")]
void same_index(uint s : SV_PrimitiveID, // expected-note {{previous use is here}}
                uint t : SV_PrimitiveID0) {} // expected-note {{'t' declared here}}
// expected-error@-1 {{semantic index overlap SV_PrimitiveID0}}

// Integer inputs can only use nointerpolation.
[shader("pixel")]
void bad_interpolation(sample uint s : SV_PrimitiveID) {}
// expected-error@-1 {{interpolation modifiers other than 'nointerpolation' cannot be used with type 'uint' (aka 'unsigned int')}}
