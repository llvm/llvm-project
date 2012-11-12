// RUN: %clang_cc1 -fsyntax-only -Wunused-parameter -Wused-but-marked-unused -Wunused -verify %s

void f1(void) {
  int x, y, z;
  #pragma unused(x)
  #pragma unused(y, z)
  
  int w; // expected-warning {{unused}}
  #pragma unused w // expected-warning{{missing '(' after '#pragma unused' - ignoring}}
}

void f2(void) {
  int x, y; // expected-warning {{unused}} expected-warning {{unused}}
  #pragma unused(x,) // expected-warning{{expected '#pragma unused' argument to be a variable name}}
  #pragma unused() // expected-warning{{expected '#pragma unused' argument to be a variable name}}
}

void f3(void) {
  #pragma unused(x) // expected-warning{{undeclared variable 'x' used as an argument for '#pragma unused'}}
}

void f4(void) {
  int w; // expected-warning {{unused}}
  #pragma unused((w)) // expected-warning{{expected '#pragma unused' argument to be a variable name}}
}

void f6(void) {
  int z; // no-warning
  {
    #pragma unused(z) // no-warning
  }
}

void f7() {
  int y;
  #pragma unused(undeclared, undefined, y) // expected-warning{{undeclared variable 'undeclared' used as an argument for '#pragma unused'}} expected-warning{{undeclared variable 'undefined' used as an argument for '#pragma unused'}}
}

int f8(int x) { // expected-warning{{unused parameter 'x'}}
  return 0;
}

int f9(int x) {
  return x;
}

int f10(int x) {
  #pragma unused(x)
  return 0;
}

int f11(int x) {
  #pragma unused(x)
  return x; // expected-warning{{'x' was marked unused but was used}}
}

int f12(int x) {
  int y = x;
  #pragma unused(x) // expected-warning{{'x' was marked unused but was used}}
  return y;
}

// rdar://8793832
static int glob_var = 0;
#pragma unused(glob_var)
