// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -Wconversion -verify %s

void literal_assignments() {
  half h;

  h = 2.0h; // No conversion, no diagnostic expected.

  // Literal conversions that don't lose precision also don't cause diagnostics.
  // Conversion from double (no diagnostic expected)
  h = 2.0l;
  h = 2.0;
  h = 2.0f;

  // Literal assignments with conversions that lose precision produce
  // diagnostics under `-Wconversion`.

  // Lose precision on assignment.
  h = 3.1415926535897932384626433h; // No diagnostic expected because this isn't a conversion.

  // Lose precision on assignment converting float to half.
  h = 3.1415926535897932384626433f; // expected-warning {{implicit conversion loses floating-point precision: 'float' to 'half'}}

  // Lose precision on assignment converting float to half.
  h = 3.1415926535897932384626433f * 2.0f; // expected-warning {{implicit conversion loses floating-point precision: 'float' to 'half'}}

  // Lose precision on assignment converting double to half.
  h = 3.1415926535897932384626433l; // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'half'}}

  // Lose precision on assignment converting double to half.
  h = 3.1415926535897932384626433l * 2.0l; // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'half'}}

  // Literal assinments of values out of the representable range produce
  // warnings.

  h = 66000.h; // expected-warning {{magnitude of floating-point constant too large for type 'half'; maximum is 65504}}
  h = -66000.h; // expected-warning {{magnitude of floating-point constant too large for type 'half'; maximum is 65504}}

  // The `h` suffix is invalid on integer literals.
  h = 66000h; // expected-error {{invalid suffix 'h' on integer constant}}
}

template <typename T, typename U>
struct is_same {
  static const bool value = false;
};

template <typename T>
struct is_same<T, T> {
  static const bool value = true;
};

_Static_assert(is_same<float, __decltype(1.0)>::value, "1.0 literal is float");

_Static_assert(is_same<half, __decltype(1.0h)>::value, "1.0h literal is half");
_Static_assert(is_same<float, __decltype(1.0f)>::value, "1.0f literal is float");
_Static_assert(is_same<double, __decltype(1.0l)>::value, "1.0l literal is double");
