// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c99

int f3(y, x,       // expected-warning {{a function declaration without a prototype is deprecated in all versions of C and is not supported in C2x}}
       x)          // expected-error {{redefinition of parameter}}
  int y, 
      x,           // expected-note {{previous declaration is here}}
      x;           // expected-error {{redefinition of parameter}}
{
  return x + y; 
} 

void f4(void) { 
  f3 (1, 1, 2, 3, 4); // expected-warning{{too many arguments}}
}

