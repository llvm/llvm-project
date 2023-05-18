// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -fno-signed-char

int a(int* x); // expected-note{{passing argument to parameter 'x' here}}
int b(unsigned* y) { return a(y); } // expected-warning {{passing 'unsigned int *' to parameter of type 'int *' converts between pointers to integer types with different sign}}

signed char *plainCharToSignedChar(signed char *arg) { // expected-note{{passing argument to parameter}}
  extern char c;
  signed char *p = &c; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  struct { signed char *p; } s = { &c }; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  p = &c; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  plainCharToSignedChar(&c); // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  return &c; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
}

char *unsignedCharToPlainChar(char *arg) { // expected-note{{passing argument to parameter}}
  extern unsigned char uc[];
  char *p = uc; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  (void) (char *[]){ [42] = uc }; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  p = uc; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  unsignedCharToPlainChar(uc); // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  return uc; // expected-warning {{converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
}
