// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -finclude-default-header -verify %s

// This test verifies that the implicit conversion warning for floating-point
// precision loss works correctly for matrix types. The IsSameFloatAfterCast
// function is used to suppress warnings for constant values that are exactly
// representable in the target type.

void TakesHalf2x2(half2x2 H);
void TakesFloat2x2(float2x2 F);

// Test 1: Non-constant matrix values should warn when precision is lost

void TestNonConstantPrecisionLoss(double2x2 D, float2x2 F) {
  // Passing double matrix to float parameter - should warn
  TakesFloat2x2(D);
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'double2x2' (aka 'matrix<double, 2, 2>') to 'float2x2' (aka 'matrix<float, 2, 2>')}}

  // Passing float matrix to half parameter - should warn
  TakesHalf2x2(F);
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'float2x2' (aka 'matrix<float, 2, 2>') to 'half2x2' (aka 'matrix<half, 2, 2>')}}

  // Passing double matrix to half parameter - should warn
  TakesHalf2x2(D);
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'double2x2' (aka 'matrix<double, 2, 2>') to 'half2x2' (aka 'matrix<half, 2, 2>')}}
}

// Test 2: Constant matrix values that are exactly representable should NOT warn

void TestConstantExactlyRepresentable() {
  // These values are exactly representable in all float types
  constexpr double2x2 D_exact = {1.0, 2.0, 3.0, 4.0};
  constexpr float2x2 F_exact = {1.0, 2.0, 3.0, 4.0};

  // Double to float with exactly representable values - no warning expected
  TakesFloat2x2(D_exact);

  // Float to half with exactly representable values - no warning expected
  TakesHalf2x2(F_exact);

  // Double to half with exactly representable values - no warning expected
  TakesHalf2x2(D_exact);
}

// Test 3: Constant matrix values that are NOT exactly representable SHOULD warn
// Note: The precision loss check compares if the value survives a round-trip
// conversion (source -> target -> source).

void TestConstantNotExactlyRepresentable() {
  constexpr float2x2 F_inexact = {0.1f, 0.2f, 0.3f, 0.4f};
  constexpr double2x2 D_inexact = {0.1l, 0.2l, 0.3l, 0.4l};
  constexpr double2x2 D_inexact2 = {0.1, 0.2, 0.3, 0.4};

  // Float to half with values that lose precision - should warn
  TakesHalf2x2(F_inexact);
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'const float2x2' (aka 'matrix<float const, 2, 2>') to 'half2x2' (aka 'matrix<half, 2, 2>')}}

  // Double to half with values that lose precision - should warn
  TakesHalf2x2(D_inexact);
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'const double2x2' (aka 'matrix<double const, 2, 2>') to 'half2x2' (aka 'matrix<half, 2, 2>')}}

  // Double to float with values that lose precision - should warn
  TakesFloat2x2(D_inexact);
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'const double2x2' (aka 'matrix<double const, 2, 2>') to 'float2x2' (aka 'matrix<float, 2, 2>')}}

  // Double to float, where the doubles are float literals that were upcast to double - should not warn
  TakesFloat2x2(D_inexact2);
  // No expected warning because the floating point literals are upcast to double. As a result, D_inexact2 survives the double -> float -> double round-trip.
}

// Test 4: Assignment with precision loss

void TestAssignmentPrecisionLoss(double2x2 D, float2x2 F) {
  float2x2 f2x2 = D;
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'double2x2' (aka 'matrix<double, 2, 2>') to 'float2x2' (aka 'matrix<float, 2, 2>')}}

  half2x2 h2x2 = F;
  // expected-warning@-1{{implicit conversion loses floating-point precision: 'float2x2' (aka 'matrix<float, 2, 2>') to 'half2x2' (aka 'matrix<half, 2, 2>')}}
}

// Test 5: No warning for promotions (increasing precision)

void TestPromotions(half2x2 H, float2x2 F) {
  // Promotion from half to float - no warning
  TakesFloat2x2(H);

  // Promotion from half to double
  double2x2 D = H;

  // Promotion from float to double
  double2x2 D2 = F;
}
