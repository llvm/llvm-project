// RUN: %clang_cc1 -fsyntax-only -verify=silent %s
// RUN: %clang_cc1 -fsyntax-only -verify=cpp -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wno-comment -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wno-comment -std=c89 -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wno-comment -std=c99 -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wno-comment -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wno-comment -std=c17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wno-comment -std=c2x -verify=silent %s

// silent-no-diagnostics

// Reject definitions in __builtin_offsetof
// https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2350.htm
int simple(void) {
  return __builtin_offsetof(struct A // cpp-error {{'A' cannot be defined in a type specifier}} \
                                        expected-warning {{defining a type within '__builtin_offsetof' is a C23 extension}}
  {
    int a;
    struct B // expected-warning {{defining a type within '__builtin_offsetof' is a C23 extension}}
    {
      int c;
      int d;
    } x;
  }, a);
}

int anonymous_struct(void) {
  return __builtin_offsetof(struct // cpp-error-re {{'(unnamed struct at {{.*}})' cannot be defined in a type specifier}} \
                                      expected-warning {{defining a type within '__builtin_offsetof' is a C23 extension}}
  {
    int a;
    int b;
  }, a);
}

int struct_in_second_param(void) {
  struct A {
    int a, b;
    int x[20];
  };
  return __builtin_offsetof(struct A, x[sizeof(struct B{int a;})]); // cpp-error {{'B' cannot be defined in a type specifier}}
}


#define offsetof(TYPE, MEMBER) __builtin_offsetof(TYPE, MEMBER)


int macro(void) {
  return offsetof(struct A // cpp-error {{'A' cannot be defined in a type specifier}} \
                              expected-warning 2 {{defining a type within 'offsetof' is a C23 extension}}
  {
    int a;
    struct B // verifier seems to think the error is emitted by the macro
             // In fact the location of the error is "B" on the line above
    {
      int c;
      int d;
    } x;
  }, a);
}

#undef offsetof

#define offsetof(TYPE, MEMBER) (&((TYPE *)0)->MEMBER)

// no warning for traditional offsetof as a function-like macro
int * macro_func(void) {
  return offsetof(struct A // cpp-error {{'A' cannot be defined in a type specifier}}
  {
    int a;
    int b;
  }, a);
}
