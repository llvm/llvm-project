// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void f() {
  char a1[] = "a"; // No error.
  char a2[] = u8"a"; // No error.
  char a3[] = u"a"; // expected-error{{initializing char array with wide string literal}}
  char a4[] = U"a"; // expected-error{{initializing char array with wide string literal}}
  char a5[] = L"a"; // expected-error{{initializing char array with wide string literal}}

  wchar_t b1[] = "a"; // expected-error{{initializing wide char array with non-wide string literal}}
  wchar_t b2[] = u8"a"; // expected-error{{initializing wide char array with non-wide string literal}}
  wchar_t b3[] = u"a"; // expected-error{{initializing wide char array with incompatible wide string literal}}
  wchar_t b4[] = U"a"; // expected-error{{initializing wide char array with incompatible wide string literal}}
  wchar_t b5[] = L"a"; // No error.

  char16_t c1[] = "a"; // expected-error{{initializing wide char array with non-wide string literal}}
  char16_t c2[] = u8"a"; // expected-error{{initializing wide char array with non-wide string literal}}
  char16_t c3[] = u"a"; // No error.
  char16_t c4[] = U"a"; // expected-error{{initializing wide char array with incompatible wide string literal}}
  char16_t c5[] = L"a"; // expected-error{{initializing wide char array with incompatible wide string literal}}

  char32_t d1[] = "a"; // expected-error{{initializing wide char array with non-wide string literal}}
  char32_t d2[] = u8"a"; // expected-error{{initializing wide char array with non-wide string literal}}
  char32_t d3[] = u"a"; // expected-error{{initializing wide char array with incompatible wide string literal}}
  char32_t d4[] = U"a"; // No error.
  char32_t d5[] = L"a"; // expected-error{{initializing wide char array with incompatible wide string literal}}

  int e1[] = "a"; // expected-error{{array initializer must be an initializer list}}
  int e2[] = u8"a"; // expected-error{{array initializer must be an initializer list}}
  int e3[] = u"a"; // expected-error{{array initializer must be an initializer list}}
  int e4[] = U"a"; // expected-error{{array initializer must be an initializer list}}
  int e5[] = L"a"; // expected-error{{array initializer must be an initializer list}}
}

void g() {
  char a[] = 1; // expected-error{{array initializer must be an initializer list or string literal}}
  wchar_t b[] = 1; // expected-error{{array initializer must be an initializer list or wide string literal}}
  char16_t c[] = 1; // expected-error{{array initializer must be an initializer list or wide string literal}}
  char32_t d[] = 1; // expected-error{{array initializer must be an initializer list or wide string literal}}
}
