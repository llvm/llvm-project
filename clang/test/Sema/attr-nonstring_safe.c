// RUN: %clang_cc1 -fsyntax-only -verify -Wunterminated-string-initialization %s -x c
// RUN: %clang_cc1 -fsyntax-only -verify -Wunterminated-string-initialization %s -x c++


// In C, the following examples are fine:
#if __cplusplus
char foo[3] = "fo\0"; // expected-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}

struct S {
  char buf[3];
  char fub[3];
} s = { "ba\0", "bo\0" }; // expected-error 2{{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}

signed char scfoo[3] = "fo\0"; // expected-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}
unsigned char ucfoo[3] = "fo\0"; // expected-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}

#else
//expected-no-diagnostics
char foo[3] = "fo\0";

struct S {
  char buf[3];
  char fub[3];
} s = { "ba\0", "bo\0" };

signed char scfoo[3] = "fo\0";
unsigned char ucfoo[3] = "fo\0";
#endif
