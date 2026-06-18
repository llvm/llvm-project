// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm-only -disable-llvm-passes -finclude-default-header -verify %s

Texture2D<float4> t;

template<class T>
float4 foo(T t) {
  int2 c = {0,0};
  return t[c];
}

[shader("pixel")]
float4 test_mips() : SV_Target {
  // expected-error@+4 {{'mips_type' is a private member of 'hlsl::Texture2D<>'}}
  // expected-note@*:* {{implicitly declared private here}}
  // expected-error@+2 {{calling a protected constructor of class 'hlsl::Texture2D<>::mips_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  Texture2D<float4>::mips_type a; 

  // expected-error@+4 {{'mips_slice_type' is a private member of 'hlsl::Texture2D<>'}}
  // expected-note@*:* {{implicitly declared private here}}
  // expected-error@+2 {{calling a protected constructor of class 'hlsl::Texture2D<>::mips_slice_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  Texture2D<float4>::mips_slice_type b;

  // expected-warning@+3 {{'auto' type specifier is a HLSL 202y extension}}
  // expected-error@+2 {{calling a protected constructor of class 'hlsl::Texture2D<>::mips_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  auto c = t.mips;

  // expected-error@+2 {{calling a protected constructor of class 'hlsl::Texture2D<>::mips_slice_type'}}
  // expected-note@*:* {{implicitly declared protected here}}
  return t.mips[0][int2(0, 0)] + foo(t.mips[0]);
}
