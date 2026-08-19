// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=RWTexture2D -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -finclude-default-header -DTEXTURE=RWTexture2DArray -verify %s

// RWTexture2D has no mips view (contrast with Texture2D-mips-errors.hlsl,
// which exercises private/protected mips_type on Texture2D).

TEXTURE<float4> t;

void test_no_mips_member() {
  // expected-error@+1 {{no member named 'mips' in 'hlsl::RWTexture2D}}
  (void)t.mips;
}

void test_no_mips_type() {
  // expected-error@+1 {{no type named 'mips_type' in 'hlsl::RWTexture2D}}
  TEXTURE<float4>::mips_type a;

  // expected-error@+1 {{no type named 'mips_slice_type' in 'hlsl::RWTexture2D}}
  TEXTURE<float4>::mips_slice_type b;
}
