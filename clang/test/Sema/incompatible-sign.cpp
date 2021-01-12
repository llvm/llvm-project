// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -fno-signed-char

void plainToSigned() {
  extern char c;
  signed char *p;
  p = &c; // expected-error {{converts between pointers to integer types with different sign}}
}

void unsignedToPlain() {
  extern unsigned char uc;
  char *p;
  p = &uc; // expected-error {{converts between pointers to integer types with different sign}}
}
