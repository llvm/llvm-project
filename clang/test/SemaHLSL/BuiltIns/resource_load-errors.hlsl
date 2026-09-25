// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -x hlsl \
// RUN:   -finclude-default-header -fnative-half-type -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify %s

// Texture2D<float4>
using tex_t = __hlsl_resource_t [[hlsl::resource_class("SRV")]]
    [[hlsl::contained_type(float4)]] [[hlsl::dimension("2D")]];
// Texture2DMS<float4>
using tex_ms_t = __hlsl_resource_t [[hlsl::resource_class("SRV")]]
    [[hlsl::contained_type(float4)]] [[hlsl::dimension("2D")]]
    [[hlsl::is_ms]];

export void load(tex_t t, int3 loc, vector<int64_t,3> wide_loc, uint3 uloc,
                 float3 floc, int2 off, uint2 uoff, float2 foff) {
  __builtin_hlsl_resource_load_level(t, loc);
  __builtin_hlsl_resource_load_level(t, loc, off);

  // expected-error@+1 {{passing 'vector<int64_t, 3>' (vector of 3 'int64_t' values) to parameter of incompatible type 'vector<int, 3>'}}
  __builtin_hlsl_resource_load_level(t, wide_loc);

  // expected-error@+1 {{passing 'uint3' (aka 'vector<uint, 3>') to parameter of incompatible type 'vector<int, 3>'}}
  __builtin_hlsl_resource_load_level(t, uloc);

  // expected-error@+1 {{passing 'float3' (aka 'vector<float, 3>') to parameter of incompatible type 'vector<int, 3>'}}
  __builtin_hlsl_resource_load_level(t, floc);

  // expected-error@+1 {{passing 'uint2' (aka 'vector<uint, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_load_level(t, loc, uoff);

  // expected-error@+1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_load_level(t, loc, foff);
}

export void load_ms(tex_ms_t t, int2 loc, int sample, int64_t lsample,
                    float fsample, int2 off, uint2 uoff, float2 foff) {
  __builtin_hlsl_resource_load_ms(t, loc, sample);
  __builtin_hlsl_resource_load_ms(t, loc, 0, off);

  // expected-error@+1 {{passing 'int64_t' (aka 'long') to parameter of incompatible type 'int'}}
  __builtin_hlsl_resource_load_ms(t, loc, lsample);

  // Names 'float', not the 'double' that default argument promotion would have
  // widened it to without CustomTypeChecking.
  // expected-error@+1 {{passing 'float' to parameter of incompatible type 'int'}}
  __builtin_hlsl_resource_load_ms(t, loc, fsample);

  // expected-error@+1 {{passing 'uint2' (aka 'vector<uint, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_load_ms(t, loc, 0, uoff);

  // expected-error@+1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_load_ms(t, loc, 0, foff);
}
