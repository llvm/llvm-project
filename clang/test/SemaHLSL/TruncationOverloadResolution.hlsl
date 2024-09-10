// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -fsyntax-only -Wconversion %s -DERROR=1 -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -ast-dump %s | FileCheck %s

// Case 1: Prefer conversion over exact match truncation.

void Half4Float2(float2 D);
void Half4Float2(half4 D);

void Case1(float4 F, double4 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half4)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half4)' lvalue Function {{.*}} 'Half4Float2' 'void (half4)'
  Half4Float2(F); // expected-warning{{implicit conversion loses floating-point precision: 'float4' (aka 'vector<float, 4>') to 'vector<half, 4>' (vector of 4 'half' values)}}
}

// Case 2: Prefer promotions over conversions when truncating.
void Half2Double2(double2 D);
void Half2Double2(half2 H);

void Case2(float4 F) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'Half2Double2' 'void (double2)'
  Half2Double2(F); // expected-warning{{implicit conversion truncates vector: 'float4' (aka 'vector<float, 4>') to 'vector<double, 2>' (vector of 2 'double' values)}}
}

#if ERROR
// Case 3: Two promotions or two conversions are ambiguous.
void Float2Double2(double2 D); // expected-note{{candidate function}}
void Float2Double2(float2 D); // expected-note{{candidate function}}

void Half2Float2(float2 D); // expected-note{{candidate function}}
void Half2Float2(half2 D); // expected-note{{candidate function}}

void Half2Half3(half3 D); // expected-note{{candidate function}} expected-note{{candidate function}} expected-note{{candidate function}}
void Half2Half3(half2 D); // expected-note{{candidate function}} expected-note{{candidate function}} expected-note{{candidate function}}

void Double2Double3(double3 D); // expected-note{{candidate function}} expected-note{{candidate function}} expected-note{{candidate function}}
void Double2Double3(double2 D); // expected-note{{candidate function}} expected-note{{candidate function}} expected-note{{candidate function}}

void Half4Float4Double2(double2 D);
void Half4Float4Double2(float4 D); // expected-note{{candidate function}}
void Half4Float4Double2(half4 D); // expected-note{{candidate function}}

void Case1(half4 H, float4 F, double4 D) {
  Half4Float4Double2(D); // expected-error {{call to 'Half4Float4Double2' is ambiguous}}

  Float2Double2(H); // expected-error {{call to 'Float2Double2' is ambiguous}}

  Half2Float2(D); // expected-error {{call to 'Half2Float2' is ambiguous}}

  Half2Half3(H); // expected-error {{call to 'Half2Half3' is ambiguous}}
  Half2Half3(F); // expected-error {{call to 'Half2Half3' is ambiguous}}
  Half2Half3(D); // expected-error {{call to 'Half2Half3' is ambiguous}}
  Half2Half3(H.xyz);
  Half2Half3(F.xyz); // expected-warning {{implicit conversion loses floating-point precision: 'vector<float, 3>' (vector of 3 'float' values) to 'vector<half, 3>' (vector of 3 'half' values)}}
  Half2Half3(D.xyz); // expected-warning {{implicit conversion loses floating-point precision: 'vector<double, 3>' (vector of 3 'double' values) to 'vector<half, 3>' (vector of 3 'half' values)}}

  Double2Double3(H); // expected-error {{call to 'Double2Double3' is ambiguous}}
  Double2Double3(F); // expected-error {{call to 'Double2Double3' is ambiguous}}
  Double2Double3(D); // expected-error {{call to 'Double2Double3' is ambiguous}}
  Double2Double3(D.xyz);
  Double2Double3(F.xyz);
  Double2Double3(H.xyz);
}
#endif
