// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -emit-llvm-only -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture2D -DINDEX_TYPE=int2 -DHAS_MIPS -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -emit-llvm-only -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=Texture2DArray -DINDEX_TYPE=int3 -DHAS_MIPS -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -emit-llvm-only -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=RWTexture2D -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -emit-llvm-only -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=RWTexture2DArray -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -emit-llvm-only -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=TextureCube -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl \
// RUN:   -emit-llvm-only -disable-llvm-passes -finclude-default-header \
// RUN:   -DTEXTURE=TextureCubeArray -verify %s

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   INDEX_TYPE         operator[] index type
//   HAS_MIPS           defined for types that have a `mips` view
//
// A texture with a mips view exposes mips_type / mips_slice_type as private

TEXTURE<float4> t;

#ifdef HAS_MIPS
template<class T>
float4 foo(T t) {
  INDEX_TYPE c = (INDEX_TYPE)0;
  return t[c];
}
#endif

[shader("pixel")]
float4 test_mips() : SV_Target {
#ifdef HAS_MIPS
  // expected-error@+4 {{'mips_type' is a private member of 'hlsl::Texture}}
  // expected-note@*:* {{implicitly declared private here}}
  // expected-error-re@+2 {{calling a protected constructor of class 'hlsl::Texture{{.*}}::mips_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  TEXTURE<float4>::mips_type a;

  // expected-error@+4 {{'mips_slice_type' is a private member of 'hlsl::Texture}}
  // expected-note@*:* {{implicitly declared private here}}
  // expected-error-re@+2 {{calling a protected constructor of class 'hlsl::Texture{{.*}}::mips_slice_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  TEXTURE<float4>::mips_slice_type b;

  // expected-warning@+3 {{'auto' type specifier is a HLSL 202y extension}}
  // expected-error-re@+2 {{calling a protected constructor of class 'hlsl::Texture{{.*}}::mips_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  auto c = t.mips;

  // expected-error-re@+2 {{calling a protected constructor of class 'hlsl::Texture{{.*}}::mips_slice_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  return t.mips[0][(INDEX_TYPE)0] + foo(t.mips[0]);
#else
  // expected-error-re@+1 {{no member named 'mips' in 'hlsl::{{.*}}Texture}}
  (void)t.mips;

  // expected-error-re@+1 {{no type named 'mips_type' in 'hlsl::{{.*}}Texture}}
  TEXTURE<float4>::mips_type a;

  // expected-error-re@+1 {{no type named 'mips_slice_type' in 'hlsl::{{.*}}Texture}}
  TEXTURE<float4>::mips_slice_type b;

  return 0;
#endif
}
