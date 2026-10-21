// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Type, index and interpolation checks for SV_PrimitiveID. These are
// independent of the shader stage and IO direction, which are covered by
// primitiveid.stages.hlsl.

[shader("pixel")]
void valid_uint(uint s : SV_PrimitiveID) {}
[shader("pixel")]
void valid_int(int s : sv_primitiveid0) {}
[shader("pixel")]
void valid_flat(nointerpolation uint s : SV_PrimitiveID) {}

// 16-bit integers are allowed. SPIR-V converts them to and from the 32-bit
// built-in.
[shader("pixel")]
void uint16(uint16_t s : SV_PrimitiveID) {}
[shader("pixel")]
void int16(int16_t s : SV_PrimitiveID) {}

struct Input {
  float4 p : SV_Position;
  uint s : SV_PrimitiveID;
};
[shader("pixel")]
void valid_struct(Input i) {}

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
