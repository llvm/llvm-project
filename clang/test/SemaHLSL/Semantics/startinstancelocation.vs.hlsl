// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -finclude-default-header -fsyntax-only -verify %s

// Type and index checks for SV_StartInstanceLocation. These are independent of
// the shader stage and IO direction, which are covered by
// startinstancelocation.stages.hlsl.

[shader("vertex")]
void valid_int(int v : SV_StartInstanceLocation) {}
[shader("vertex")]
void valid_uint(uint v : sv_startinstancelocation0) {}

struct Input {
  int v : SV_StartInstanceLocation;
  float4 p : POSITION;
};
[shader("vertex")]
void valid_struct(Input i) {}

[shader("vertex")]
void bad_float(float v : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' must be a scalar of 32 bit integer type (was 'float')}}
[shader("vertex")]
void bad_bool(bool v : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' must be a scalar of 32 bit integer type (was 'bool')}}
[shader("vertex")]
void bad_int16(int16_t v : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' must be a scalar of 32 bit integer type (was 'int16_t' (aka 'short'))}}
[shader("vertex")]
void bad_int64(int64_t v : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' must be a scalar of 32 bit integer type (was 'int64_t' (aka 'long'))}}
[shader("vertex")]
void bad_vector(uint1 v : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' must be a scalar of 32 bit integer type (was 'uint1' (aka 'vector<uint, 1>'))}}
[shader("vertex")]
void bad_matrix(int1x1 v : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' must be a scalar of 32 bit integer type (was 'int1x1' (aka 'matrix<int, 1, 1>'))}}
[shader("vertex")]
void bad_array(int v[1] : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' must be a scalar of 32 bit integer type (was 'int[1]')}}

[shader("vertex")]
void bad_index(int v : SV_StartInstanceLocation1) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' does not allow indexing}}

[shader("vertex")]
void same_index(int v : SV_StartInstanceLocation, // expected-note {{previous use is here}}
                int w : SV_StartInstanceLocation0) {} // expected-note {{'w' declared here}}
// expected-error@-1 {{semantic index overlap SV_StartInstanceLocation0}}
