// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm-only -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2D -DCOORD=int2 -DZEROS="0, 0" -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm-only -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DArray -DCOORD=int3 -DZEROS="0, 0, 0" -verify %s

// The diagnostics are truncated to 'hlsl::Texture' (dropping the dimension
// suffix such as "2D<>") so the same assertions apply to every texture type,
// while still asserting the mips_type / mips_slice_type member. The
// protected-constructor checks use `-re` with `{{.*}}` to bridge 'hlsl::Texture'
// and the trailing member name. RWTexture2D/RWTexture2DArray have no mips member
// and are not tested here.

TEXTURE<float4> t;

template<class T>
float4 foo(T t) {
  COORD c = {ZEROS};
  return t[c];
}

[shader("pixel")]
float4 test_mips() : SV_Target {
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
  return t.mips[0][COORD(ZEROS)] + foo(t.mips[0]);
}
