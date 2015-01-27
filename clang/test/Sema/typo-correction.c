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
