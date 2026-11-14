// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Invalid type, index and interpolation uses of SV_IsFrontFace. The valid
// cases are covered by isfrontface.ps.hlsl, and the shader stage and IO
// direction by isfrontface.stages.hlsl.

[shader("pixel")]
void bad_float(float s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'float')}}
[shader("pixel")]
void bad_int64(uint64_t s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'uint64_t' (aka 'unsigned long')}}
[shader("pixel")]
void bad_vector(bool3 s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'bool3' (aka 'vector<bool, 3>'))}}
[shader("pixel")]
void bad_vector1(bool1 s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'bool1' (aka 'vector<bool, 1>'))}}
[shader("pixel")]
void bad_matrix(bool2x2 s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'bool2x2' (aka 'matrix<bool, 2, 2>'))}}
// expected-error@-2 {{semantic 'SV_IsFrontFace' does not allow indexing}}
// Only 32-bit integers are allowed, so 16-bit types are rejected.
[shader("pixel")]
void bad_uint16(uint16_t s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'uint16_t' (aka 'unsigned short')}}
[shader("pixel")]
void bad_int16(int16_t s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'int16_t' (aka 'short')}}
[shader("pixel")]
void bad_half(half s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'half')}}
[shader("pixel")]
void bad_array(bool s[1] : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'bool[1]'}}
[shader("pixel")]
void bad_array2(bool s[2] : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' must be a scalar of bool or 32 bit integer type (was 'bool[2]'}}
// expected-error@-2 {{semantic 'SV_IsFrontFace' does not allow indexing}}

[shader("pixel")]
void bad_index(bool s : SV_IsFrontFace1) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' does not allow indexing}}

[shader("pixel")]
void same_index(bool s : SV_IsFrontFace, // expected-note {{previous use is here}}
                bool t : SV_IsFrontFace0) {} // expected-note {{'t' declared here}}
// expected-error@-1 {{semantic index overlap SV_IsFrontFace0}}

// Integer and bool inputs can only use nointerpolation.
[shader("pixel")]
void bad_interpolation_uint(sample uint s : SV_IsFrontFace) {}
// expected-error@-1 {{interpolation modifiers other than 'nointerpolation' cannot be used with type 'uint' (aka 'unsigned int')}}
[shader("pixel")]
void bad_interpolation_bool(linear bool s : SV_IsFrontFace) {}
// expected-error@-1 {{interpolation modifiers other than 'nointerpolation' cannot be used with type 'bool'}}
