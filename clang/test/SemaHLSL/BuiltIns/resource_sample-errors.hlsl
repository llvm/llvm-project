// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -x hlsl \
// RUN:   -finclude-default-header -fnative-half-type -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify %s

// Texture2D<float4>
using tex_t = __hlsl_resource_t [[hlsl::resource_class("SRV")]]
    [[hlsl::contained_type(float4)]] [[hlsl::dimension("2D")]];

using samp_t = __hlsl_resource_t [[hlsl::resource_class("Sampler")]];
using samp_cmp_t = __hlsl_resource_t [[hlsl::resource_class("Sampler")]];

//
// Sample location.
//

export void location(tex_t t, samp_t s, float2 uv, double2 duv, half2 huv,
                     int2 iuv, float3 uvw, float u) {
  __builtin_hlsl_resource_sample(t, s, uv);

  // expected-error@+1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_sample(t, s, duv);

  // expected-error@+1 {{passing 'half2' (aka 'vector<half, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_sample(t, s, huv);

  // expected-error@+1 {{passing 'int2' (aka 'vector<int, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_sample(t, s, iuv);

  // expected-error@+1 {{passing 'float3' (aka 'vector<float, 3>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_sample(t, s, uvw);

  // expected-error@+1 {{passing 'float' to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_sample(t, s, u);
}

//
// Offset.
//

export void offset(tex_t t, samp_t s, float2 uv, int2 off, uint2 uoff,
                   float2 foff, vector<int64_t,2> loff) {
  __builtin_hlsl_resource_sample(t, s, uv, off);

  // expected-error@+1 {{passing 'uint2' (aka 'vector<uint, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_sample(t, s, uv, uoff);

  // expected-error@+1 {{passing 'vector<int64_t, 2>' (vector of 2 'int64_t' values) to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_sample(t, s, uv, loff);

  // expected-error@+1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_sample(t, s, uv, foff);
}

//
// Scalar float operands: bias, LOD, compare value and clamp.
//

export void scalar_operands(tex_t t, samp_t s, samp_cmp_t sc, float2 uv,
                            float f, double d, half h, float2 v) {
  // Each is a single 32-bit float.
  __builtin_hlsl_resource_sample_bias(t, s, uv, f);

  // expected-error@+1 {{passing 'double' to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample_bias(t, s, uv, d);

  // expected-error@+1 {{passing 'half' to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample_bias(t, s, uv, h);

  // expected-error@+1 {{passing 'double' to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample_level(t, s, uv, d);

  // expected-error@+1 {{passing 'double' to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample_cmp(t, sc, uv, d);

  // expected-error@+1 {{passing 'double' to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample_cmp_level_zero(t, sc, uv, d);

  // expected-error@+1 {{passing 'double' to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample(t, s, uv, int2(0, 0), d);

  // expected-error@+1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample(t, s, uv, int2(0, 0), v);

  // expected-error@+1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_sample_cmp_level_zero(t, sc, uv, v);
}

//
// Gradients.
//

export void gradients(tex_t t, samp_t s, float2 uv, double2 d, float3 wide) {
  __builtin_hlsl_resource_sample_grad(t, s, uv, uv, uv);

  // expected-error@+1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_sample_grad(t, s, uv, d, uv);

  // expected-error@+1 {{passing 'float3' (aka 'vector<float, 3>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_sample_grad(t, s, uv, wide, uv);
}

//
// CalculateLevelOfDetail.
//

export void calculate_lod(tex_t t, samp_t s, float2 uv, double2 duv) {
  __builtin_hlsl_resource_calculate_lod(t, s, uv);

  // expected-error@+1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_calculate_lod(t, s, duv);

  // expected-error@+1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_calculate_lod_unclamped(t, s, duv);

  // expected-error@+1 {{passing 'float' to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_calculate_lod(t, s, uv.x);
}
