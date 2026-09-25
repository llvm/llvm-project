// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -x hlsl \
// RUN:   -finclude-default-header -fnative-half-type -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify %s

// Texture2D<float4>
using tex_t = __hlsl_resource_t [[hlsl::resource_class("SRV")]]
    [[hlsl::contained_type(float4)]] [[hlsl::dimension("2D")]];
using samp_t = __hlsl_resource_t [[hlsl::resource_class("Sampler")]];
using samp_cmp_t = __hlsl_resource_t [[hlsl::resource_class("Sampler")]];

export void gather(tex_t t, samp_t s, samp_cmp_t sc, float2 uv, double2 duv,
                   uint comp, int64_t lcomp, float fcomp, int2 off, uint2 uoff,
                   float2 foff, float cmp, double d) {
  __builtin_hlsl_resource_gather(t, s, uv, comp);
  __builtin_hlsl_resource_gather(t, s, uv, 0u, off);

  // expected-error@+1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_gather(t, s, duv, 0u);

  // expected-error@+1 {{passing 'int' to parameter of incompatible type 'unsigned int'}}
  __builtin_hlsl_resource_gather(t, s, uv, 0);

  // expected-error@+1 {{passing 'int64_t' (aka 'long') to parameter of incompatible type 'unsigned int'}}
  __builtin_hlsl_resource_gather(t, s, uv, lcomp);

  // expected-error@+1 {{passing 'float' to parameter of incompatible type 'unsigned int'}}
  __builtin_hlsl_resource_gather(t, s, uv, fcomp);

  // expected-error@+1 {{passing 'uint2' (aka 'vector<uint, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_gather(t, s, uv, 0u, uoff);

  // expected-error@+1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type 'vector<int, 2>'}}
  __builtin_hlsl_resource_gather(t, s, uv, 0u, foff);

  __builtin_hlsl_resource_gather_cmp(t, sc, uv, cmp, 0u);

  // expected-error@+1 {{passing 'double' to parameter of incompatible type 'float'}}
  __builtin_hlsl_resource_gather_cmp(t, sc, uv, d, 0u);

  // expected-error@+1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type 'vector<float, 2>'}}
  __builtin_hlsl_resource_gather_cmp(t, sc, duv, cmp, 0u);
}
