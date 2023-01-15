// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c89 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c99 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c2x -verify %s

// Reject definitions in __builtin_offsetof
// https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2350.htm
int simple(void) {
  return __builtin_offsetof(struct A // expected-error{{'struct A' cannot be defined in '__builtin_offsetof'}} 
  { 
    int a;
    struct B // expected-error{{'struct B' cannot be defined in '__builtin_offsetof'}} 
    {
      int c;
      int d;
    } x;
  }, a);
}

int anonymous_struct() {
  return __builtin_offsetof(struct // expected-error-re{{'struct (unnamed at {{.*}})' cannot be defined in '__builtin_offsetof'}}
  { 
    int a;
    int b;
  }, a);
}

int struct_in_second_param() {
  struct A {
    int a, b;
    int x[20];
  };
  return __builtin_offsetof(struct A, x[sizeof(struct B{int a;})]); // no-error
}


#define offsetof(TYPE, MEMBER) __builtin_offsetof(TYPE, MEMBER)


int macro(void) {
  return offsetof(struct A // expected-error{{'struct A' cannot be defined in 'offsetof'}}
                           // expected-error@-1{{'struct B' cannot be defined in 'offsetof'}}
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
  return offsetof(struct A // no-warning
  { 
    int a;
    int b;
  }, a);
}
