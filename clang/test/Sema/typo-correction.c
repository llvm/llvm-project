// RUN: %clang_cc1 -fsyntax-only -verify %s
//
// This file contains typo correction tests which hit different code paths in C
// than in C++ and may exhibit different behavior as a result.

__typeof__(struct F*) var[invalid];  // expected-error-re {{use of undeclared identifier 'invalid'{{$}}}}

void PR21656() {
  float x;
  x = (float)arst;  // expected-error-re {{use of undeclared identifier 'arst'{{$}}}}
}

a = b ? : 0;  // expected-warning {{type specifier missing, defaults to 'int'}} \
              // expected-error {{use of undeclared identifier 'b'}}

int foobar;  // expected-note {{'foobar' declared here}}
a = goobar ?: 4;  // expected-warning {{type specifier missing, defaults to 'int'}} \
                  // expected-error {{use of undeclared identifier 'goobar'; did you mean 'foobar'?}} \
                  // expected-error {{initializer element is not a compile-time constant}}

struct ContainerStuct {
  enum { SOME_ENUM }; // expected-note {{'SOME_ENUM' declared here}}
};

void func(int arg) {
  switch (arg) {
  case SOME_ENUM_: // expected-error {{use of undeclared identifier 'SOME_ENUM_'; did you mean 'SOME_ENUM'}}
    ;
  }
}

void banana(void);  // expected-note {{'banana' declared here}}
int c11Generic(int arg) {
  _Generic(hello, int : banana)();  // expected-error-re {{use of undeclared identifier 'hello'{{$}}}}
  _Generic(arg, int : bandana)();  // expected-error {{use of undeclared identifier 'bandana'; did you mean 'banana'?}}
}

typedef long long __m128i __attribute__((__vector_size__(16)));
int PR23101(__m128i __x) {
  return foo((__v2di)__x);  // expected-warning {{implicit declaration of function 'foo'}} \
                            // expected-error {{use of undeclared identifier '__v2di'}}
}

void f(long *a, long b) {
      __atomic_or_fetch(a, b, c);  // expected-error {{use of undeclared identifier 'c'}}
}

extern double cabs(_Complex double z);
void fn1() {
  cabs(errij);  // expected-error {{use of undeclared identifier 'errij'}}
}

extern long afunction(int); // expected-note {{'afunction' declared here}}
void fn2() {
  f(THIS_IS_AN_ERROR, // expected-error {{use of undeclared identifier 'THIS_IS_AN_ERROR'}}
    afunction(afunction_));  // expected-error {{use of undeclared identifier 'afunction_'; did you mean 'afunction'?}}
}
