// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -Wconversion -verify -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -Wno-conversion -ast-dump %s | FileCheck %s

// This test verifies floating point type implicit conversion ranks for overload
// resolution. In HLSL the built-in type ranks are half < float < double. This
// applies to both scalar and vector types.

// HLSL allows implicit truncation fo types, so it differentiates between
// promotions (converting to larger types) and conversions (converting to
// smaller types). Promotions are preferred over conversions. Promotions prefer
// promoting to the next lowest type in the ranking order. Conversions prefer
// converting to the next highest type in the ranking order.

void HalfFloatDouble(double2 D);
void HalfFloatDouble(float2 F);
void HalfFloatDouble(half2 H);

// CHECK: FunctionDecl {{.*}} used HalfFloatDouble 'void (double2)'
// CHECK: FunctionDecl {{.*}} used HalfFloatDouble 'void (float2)'
// CHECK: FunctionDecl {{.*}} used HalfFloatDouble 'void (half2)'

void FloatDouble(double2 D);
void FloatDouble(float2 F);

// CHECK: FunctionDecl {{.*}} used FloatDouble 'void (double2)'
// CHECK: FunctionDecl {{.*}} used FloatDouble 'void (float2)'

void HalfDouble(double2 D);
void HalfDouble(half2 H);

// CHECK: FunctionDecl {{.*}} used HalfDouble 'void (double2)'
// CHECK: FunctionDecl {{.*}} used HalfDouble 'void (half2)'

void HalfFloat(float2 F);
void HalfFloat(half2 H);

// CHECK: FunctionDecl {{.*}} used HalfFloat 'void (float2)'
// CHECK: FunctionDecl {{.*}} used HalfFloat 'void (half2)'

void Double(double2 D);
void Float(float2 F);
void Half(half2 H);

// CHECK: FunctionDecl {{.*}} used Double 'void (double2)'
// CHECK: FunctionDecl {{.*}} used Float 'void (float2)'
// CHECK: FunctionDecl {{.*}} used Half 'void (half2)'

// Case 1: A function declared with overloads for half float and double types.
//   (a) When called with half, it will resolve to half because half is an exact
//   match.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case1 'void (half2, float2, double2)'
void Case1(half2 H, float2 F, double2 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half2)' lvalue Function {{.*}} 'HalfFloatDouble' 'void (half2)'
  HalfFloatDouble(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'HalfFloatDouble' 'void (float2)'
  HalfFloatDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'HalfFloatDouble' 'void (double2)'
  HalfFloatDouble(D);
}

// Case 2: A function declared with double and float overlaods.
//   (a) When called with half, it will resolve to float because float is lower
//   ranked than double.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case2 'void (half2, float2, double2)'
void Case2(half2 H, float2 F, double2 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'FloatDouble' 'void (float2)'
  FloatDouble(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'FloatDouble' 'void (float2)'
  FloatDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'FloatDouble' 'void (double2)'
  FloatDouble(D);
}

// Case 3: A function declared with half and double overloads
//   (a) When called with half, it will resolve to half because it is an exact
//   match.
//   (b) When called with flaot, it will resolve to double because double is a
//   valid promotion.
//   (c) When called with double, it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case3 'void (half2, float2, double2)'
void Case3(half2 H, float2 F, double2 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half2)' lvalue Function {{.*}} 'HalfDouble' 'void (half2)'
  HalfDouble(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'HalfDouble' 'void (double2)'
  HalfDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'HalfDouble' 'void (double2)'
  HalfDouble(D);
}

// Case 4: A function declared with half and float overloads.
//   (a) When called with half, it will resolve to half because half is an exact
//   match.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to float because it is the
//   float is higher rank than half.

// CHECK-LABEL: FunctionDecl {{.*}} Case4 'void (half2, float2, double2)'
void Case4(half2 H, float2 F, double2 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half2)' lvalue Function {{.*}} 'HalfFloat' 'void (half2)'
  HalfFloat(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'HalfFloat' 'void (float2)'
  HalfFloat(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'HalfFloat' 'void (float2)'
  HalfFloat(D); // expected-warning{{implicit conversion loses floating-point precision: 'double2' (aka 'vector<double, 2>') to 'float2' (aka 'vector<float, 2>')}}
}

// Case 5: A function declared with only a double overload.
//   (a) When called with half, it will resolve to double because double is a
//   valid promotion.
//   (b) When called with float it will resolve to double because double is a
//   valid promotion.
//   (c) When called with double it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case5 'void (half2, float2, double2)'
void Case5(half2 H, float2 F, double2 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'Double' 'void (double2)'
  Double(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'Double' 'void (double2)'
  Double(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double2)' lvalue Function {{.*}} 'Double' 'void (double2)'
  Double(D);
}

// Case 6: A function declared with only a float overload.
//   (a) When called with half, it will resolve to float because float is a
//   valid promotion.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to float because it is a
//   valid conversion.

// CHECK-LABEL: FunctionDecl {{.*}} Case6 'void (half2, float2, double2)'
void Case6(half2 H, float2 F, double2 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'Float' 'void (float2)'
  Float(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'Float' 'void (float2)'
  Float(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float2)' lvalue Function {{.*}} 'Float' 'void (float2)'
  Float(D); // expected-warning{{implicit conversion loses floating-point precision: 'double2' (aka 'vector<double, 2>') to 'float2' (aka 'vector<float, 2>')}}
}

// Case 7: A function declared with only a half overload.
//   (a) When called with half, it will resolve to half because half is an
//   exact match
//   (b) When called with float it will resolve to half because half is a
//   valid conversion.
//   (c) When called with double it will resolve to float because it is a
//   valid conversion.

// CHECK-LABEL: FunctionDecl {{.*}} Case7 'void (half2, float2, double2)'
void Case7(half2 H, float2 F, double2 D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half2)' lvalue Function {{.*}} 'Half' 'void (half2)'
  Half(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half2)' lvalue Function {{.*}} 'Half' 'void (half2)'
  Half(F); // expected-warning{{implicit conversion loses floating-point precision: 'float2' (aka 'vector<float, 2>') to 'half2' (aka 'vector<half, 2>')}}

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half2)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half2)' lvalue Function {{.*}} 'Half' 'void (half2)'
  Half(D); // expected-warning{{implicit conversion loses floating-point precision: 'double2' (aka 'vector<double, 2>') to 'half2' (aka 'vector<half, 2>')}}
}
