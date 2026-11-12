// RUN: %clang_cc1 -std=c++1y -verify %s

int operator""ms(unsigned long long); // expected-warning {{reserved}}
float operator""ms(long double); // expected-warning {{reserved}}

int operator""_foo(unsigned long long);

namespace integral {
  static_assert(1'2'3 == 12'3, "");
  static_assert(1'000'000 == 0xf'4240, "");
  static_assert(0'004'000'000 == 0x10'0000, "");
  static_assert(0b0101'0100 == 0x54, "");

  int a = 123'; //'; // expected-error {{expected ';'}}
  int b = 0'xff; // expected-error {{digit separator cannot appear at end of digit sequence}} expected-error {{suffix 'xff' on integer}}
  int c = 0x'ff; // expected-error {{suffix 'x'ff' on integer}}
  int d = 0'1234; // ok, octal
  int e = 0'b1010; // expected-error {{digit 'b' in octal constant}}
  int f = 0b'1010; // expected-error {{invalid digit 'b' in octal}}
  int g = 123'ms; // expected-error {{digit separator cannot appear at end of digit sequence}}
  int h = 0x1e+1; // expected-error {{invalid suffix '+1' on integer constant}}
  int i = 0x1'e+1; // ok, 'e+' is not recognized after a digit separator

  int z = 0'123'_foo; //'; // expected-error {{cannot appear at end of digit seq}}
}

namespace floating {
  static_assert(0'123.456'7 == 123.4567, "");
  static_assert(1e1'0 == 10'000'000'000, "");

  float a = 1'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float b = 1'0e1;
  float c = 1.'0e1; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float d = 1.0'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float e = 1e'1; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float f = 1e1'ms; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float g = 0.'0; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float h = .'0; // '; // expected-error {{expected expression}}, lexed as . followed by character literal
  float i = 0x.'0p0; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float j = 0x'0.0p0; // expected-error {{invalid suffix 'x'0.0p0'}}
  float k = 0x0'.0p0; // '; // expected-error {{expected ';'}}
  float l = 0x0.'0p0; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float m = 0x0.0'p0; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float n = 0x0.0p'0; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float o = 0x0.0p0'ms; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float p = 0'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float q = 0'0e1;
  float r = 0.'0e1; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float s = 0.0'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float t = 0.0e'1; // expected-error {{digit separator cannot appear at start of digit sequence}}
  float u = 0x.'p1f; // expected-error {{hexadecimal floating literal requires a significand}}
  float v = 0e'f; // expected-error {{exponent has no digits}}
  float w = 0x0p'f; // expected-error {{exponent has no digits}}
  float x = 0'e+1; // expected-error {{digit separator cannot appear at end of digit sequence}}
  float y = 0x0'p+1; // expected-error {{digit separator cannot appear at end of digit sequence}}
}

#line 123'456
static_assert(__LINE__ == 123456, "");

// x has value 0 in C++11 and 34 in C++1y.
#define M(x, ...) __VA_ARGS__
constexpr int x = { M(1'2,3'4) };
static_assert(x == 34, "");

namespace UCNs {
  // UCNs can appear before digit separators but not after.
  int a = 0\u1234'5; // expected-error {{invalid suffix '\u1234'5' on integer constant}}
  int b = 0'\u12345; // '; // expected-error {{expected ';'}}
  constexpr int c {M(0\u1234'0,0'1)};
  constexpr int d {M(00'\u1234,0'1)};
  static_assert(c == 1, "");
  static_assert(d == 0, "");
}

namespace UTF8 {
  // extended characters can appear before digit separators but not after.
  int a = 0ሴ'5; // expected-error {{invalid suffix 'ሴ'5' on integer constant}}
  int b = 0'ሴ5; // '; // expected-error {{expected ';'}}
  constexpr int c {M(0ሴ'0,0'1)};
  constexpr int d {M(00'ሴ,0'1)};
  static_assert(c == 1, "");
  static_assert(d == 0, "");
}
