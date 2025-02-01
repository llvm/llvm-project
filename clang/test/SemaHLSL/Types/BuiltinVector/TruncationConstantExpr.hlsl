// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s

// expected-no-diagnostics

// Note: these tests are a bit awkward because at time of writing we don't have a
// good way to constexpr `any` for bool vector conditions, and the condition for
// _Static_assert must be an integral constant.

struct S {
  int3 A;
  float B;
};

export void fn() {

  _Static_assert(((float4)(int[6]){1,2,3,4,5,6}).x == 1.0, "Woo!");

  // This compiling successfully verifies that the array constant expression
  // gets truncated to a float at compile time for instantiation via the
  // flat cast
  _Static_assert(((int)(int[2]){1,2}) == 1, "Woo!");

  // This compiling successfully verifies that the struct constant expression
  // gets truncated to an integer at compile time for instatiation via the
  // flat cast
  _Static_assert(((int)(S){{1,2,3},1.0}) == 1, "Woo!");

  _Static_assert(((float)(float[2]){1.0,2.0}) == 1.0, "Woo!");

  // This compiling successfully verifies that the vector constant expression
  // gets truncated to an integer at compile time for instantiation.
  _Static_assert(((int)1.xxxx) + 0 == 1, "Woo!");

  // This compiling successfully verifies that the vector constant expression
  // gets truncated to a float at compile time for instantiation.
  _Static_assert(((float)1.0.xxxx) + 0.0 == 1.0, "Woo!");

  // This compiling successfully verifies that a vector can be truncated to a
  // smaller vector, then truncated to a float as a constant expression.
  _Static_assert(((float2)float4(6, 5, 4, 3)).x == 6, "Woo!");
}
