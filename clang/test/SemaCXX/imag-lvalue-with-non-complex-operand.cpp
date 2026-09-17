// RUN: %clang_cc1 -fsyntax-only -verify %s

void lvalue_with_imag_int() {
  int i;
  __imag__ i = 0;   // expected-error {{expression is not assignable}}
}

void lvalue_with_imag_float() {
  float i;
  __imag__ i = 0;   // expected-error {{expression is not assignable}}
}

_Complex float foo()
{
  float f;
  __real__ f = 0;
  __imag__ f = 0;    // expected-error {{expression is not assignable}}
  return f;
}

_Complex float baz()
{
  float f;
  __real__ f = 0;
  __imag__
} // expected-error {{expected expression}}


typedef float C;
C lvalue_with_imag_float_with_typedef()
{
  C f;
  __real__ f = 0;
  __imag__ f = 0;   // expected-error {{expression is not assignable}}
  return f;
}
