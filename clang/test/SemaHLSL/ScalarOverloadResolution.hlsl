// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -Wconversion -verify -o - -DERROR=1 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -ast-dump %s | FileCheck %s

// This test verifies floating point type implicit conversion ranks for overload
// resolution. In HLSL the built-in type ranks are half < float < double. This
// applies to both scalar and vector types.

// HLSL allows implicit truncation fo types, so it differentiates between
// promotions (converting to larger types) and conversions (converting to
// smaller types). Promotions are preferred over conversions. Promotions prefer
// promoting to the next lowest type in the ranking order. Conversions prefer
// converting to the next highest type in the ranking order.

void HalfFloatDouble(double D);
void HalfFloatDouble(float F);
void HalfFloatDouble(half H);

// CHECK: FunctionDecl {{.*}} used HalfFloatDouble 'void (double)'
// CHECK: FunctionDecl {{.*}} used HalfFloatDouble 'void (float)'
// CHECK: FunctionDecl {{.*}} used HalfFloatDouble 'void (half)'

void FloatDouble(double D); // expected-note{{candidate function}}
void FloatDouble(float F); // expected-note{{candidate function}}

// CHECK: FunctionDecl {{.*}} used FloatDouble 'void (double)'
// CHECK: FunctionDecl {{.*}} used FloatDouble 'void (float)'

void HalfDouble(double D);
void HalfDouble(half H);

// CHECK: FunctionDecl {{.*}} used HalfDouble 'void (double)'
// CHECK: FunctionDecl {{.*}} used HalfDouble 'void (half)'

void HalfFloat(float F); // expected-note{{candidate function}}
void HalfFloat(half H); // expected-note{{candidate function}}

// CHECK: FunctionDecl {{.*}} used HalfFloat 'void (float)'
// CHECK: FunctionDecl {{.*}} used HalfFloat 'void (half)'

void Double(double D);
void Float(float F);
void Half(half H);

// CHECK: FunctionDecl {{.*}} used Double 'void (double)'
// CHECK: FunctionDecl {{.*}} used Float 'void (float)'
// CHECK: FunctionDecl {{.*}} used Half 'void (half)'


// Case 1: A function declared with overloads for half float and double types.
//   (a) When called with half, it will resolve to half because half is an exact
//   match.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case1 'void (half, float, double)'
void Case1(half H, float F, double D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'HalfFloatDouble' 'void (half)'
  HalfFloatDouble(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'HalfFloatDouble' 'void (float)'
  HalfFloatDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'HalfFloatDouble' 'void (double)'
  HalfFloatDouble(D);
}

// Case 2: A function declared with double and float overlaods.
//   (a) When called with half, it will fail to resolve because it cannot
//   disambiguate the promotions.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case2 'void (half, float, double)'
void Case2(half H, float F, double D) {
  #if ERROR
  FloatDouble(H); // expected-error{{call to 'FloatDouble' is ambiguous}}
  #endif

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'FloatDouble' 'void (float)'
  FloatDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'FloatDouble' 'void (double)'
  FloatDouble(D);
}

// Case 3: A function declared with half and double overloads
//   (a) When called with half, it will resolve to half because it is an exact
//   match.
//   (b) When called with flaot, it will resolve to double because double is a
//   valid promotion.
//   (c) When called with double, it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case3 'void (half, float, double)'
void Case3(half H, float F, double D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'HalfDouble' 'void (half)'
  HalfDouble(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'HalfDouble' 'void (double)'
  HalfDouble(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'HalfDouble' 'void (double)'
  HalfDouble(D);
}

// Case 4: A function declared with half and float overloads.
//   (a) When called with half, it will resolve to half because half is an exact
//   match.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to float because it is the
//   float is higher rank than half.

// CHECK-LABEL: FunctionDecl {{.*}} Case4 'void (half, float, double)'
void Case4(half H, float F, double D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'HalfFloat' 'void (half)'
  HalfFloat(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'HalfFloat' 'void (float)'
  HalfFloat(F);

  #if ERROR
  HalfFloat(D); // expected-error{{call to 'HalfFloat' is ambiguous}}
  #endif
}

// Case 5: A function declared with only a double overload.
//   (a) When called with half, it will resolve to double because double is a
//   valid promotion.
//   (b) When called with float it will resolve to double because double is a
//   valid promotion.
//   (c) When called with double it will resolve to double because it is an
//   exact match.

// CHECK-LABEL: FunctionDecl {{.*}} Case5 'void (half, float, double)'
void Case5(half H, float F, double D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'Double' 'void (double)'
  Double(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'Double' 'void (double)'
  Double(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(double)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (double)' lvalue Function {{.*}} 'Double' 'void (double)'
  Double(D);
}

// Case 6: A function declared with only a float overload.
//   (a) When called with half, it will resolve to float because float is a
//   valid promotion.
//   (b) When called with float it will resolve to float because float is an
//   exact match.
//   (c) When called with double it will resolve to float because it is a
//   valid conversion.

// CHECK-LABEL: FunctionDecl {{.*}} Case6 'void (half, float, double)'
void Case6(half H, float F, double D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'Float' 'void (float)'
  Float(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'Float' 'void (float)'
  Float(F);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (float)' lvalue Function {{.*}} 'Float' 'void (float)'
  Float(D); // expected-warning{{implicit conversion loses floating-point precision: 'double' to 'float'}}
}

// Case 7: A function declared with only a half overload.
//   (a) When called with half, it will resolve to half because half is an
//   exact match
//   (b) When called with float it will resolve to half because half is a
//   valid conversion.
//   (c) When called with double it will resolve to float because it is a
//   valid conversion.

// CHECK-LABEL: FunctionDecl {{.*}} Case7 'void (half, float, double)'
void Case7(half H, float F, double D) {
  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'Half' 'void (half)'
  Half(H);

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'Half' 'void (half)'
  Half(F); // expected-warning{{implicit conversion loses floating-point precision: 'float' to 'half'}}

  // CHECK: CallExpr {{.*}} 'void'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(half)' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'void (half)' lvalue Function {{.*}} 'Half' 'void (half)'
  Half(D); // expected-warning{{implicit conversion loses floating-point precision: 'double' to 'half'}}
}
