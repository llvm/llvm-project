// RUN: %clang_cc1 -fsyntax-only -verify %s

void lvalue_with_imag_int() {
  int i;
  __imag__ i = 0;   // expected-error {{expression is not assignable}}
}

void lvalue_with_imag_float() {
  float i;
  __imag__ i = 0;   // expected-error {{expression is not assignable}}
}

_Complex float foo() // expected-note {{previous definition is here}}
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
  __imag__ }      // expected-error {{expected expression}}


typedef _Complex float C;
C foo()          // expected-error {{redefinition of 'foo'}}
{
  C f;
  __real__ f = 0;
  __imag__ f = 0;
  return f;
}
