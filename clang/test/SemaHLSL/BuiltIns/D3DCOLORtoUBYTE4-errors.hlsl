// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

int4 test_too_few_arg() {
  return D3DCOLORtoUBYTE4();
  // expected-error@-1 {{no matching function for call to 'D3DCOLORtoUBYTE4'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires single argument 'V', but no arguments were provided}}
}

int4 test_too_many_arg(float4 v) {
  return D3DCOLORtoUBYTE4(v, v);
  // expected-error@-1 {{no matching function for call to 'D3DCOLORtoUBYTE4'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires single argument 'V', but 2 arguments were provided}}
}

int4 float2_arg(float2 v) {
    return D3DCOLORtoUBYTE4(v);
    // expected-error@-1 {{no matching function for call to 'D3DCOLORtoUBYTE4'}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: no known conversion from 'vector<[...], 2>' to 'vector<[...], 4>' for 1st argument}}
}

struct S {
  float4 f;
};

int4 struct_arg(S v) {
    return D3DCOLORtoUBYTE4(v);
    // expected-error@-1 {{no matching function for call to 'D3DCOLORtoUBYTE4'}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: no known conversion from 'S' to 'vector<float, 4>' (vector of 4 'float' values) for 1st argument}}
}
