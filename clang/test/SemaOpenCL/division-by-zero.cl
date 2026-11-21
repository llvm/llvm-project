// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -verify=expected,defined,c %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL1.2 -verify=expected,defined,c %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=clc++2021 -verify=expected,defined,cxx %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -fno-defined-int-divrem -verify=expected,undefined,c %s

// OpenCL C v3.0 s6.5.1: integer division and remainder by zero result in an
// unspecified value. The warning stays, since a zero divisor is almost always
// a bug, but it does not call the operation undefined.

void f(int x) {
  (void)(x / 0); // defined-warning {{division by zero results in an unspecified value}} \
                    undefined-warning {{division by zero is undefined}}
  (void)(x % 0); // defined-warning {{remainder by zero results in an unspecified value}} \
                    undefined-warning {{remainder by zero is undefined}}
  x /= 0;        // defined-warning {{division by zero results in an unspecified value}} \
                    undefined-warning {{division by zero is undefined}}
  x %= 0;        // defined-warning {{remainder by zero results in an unspecified value}} \
                    undefined-warning {{remainder by zero is undefined}}
}

// Constant evaluation is not changed: a division by zero is still not a
// constant expression, and INT_MIN / -1 still folds to INT_MIN.
__constant int c = 1 / 0; // c-error {{initializer element is not a compile-time constant}} \
                             defined-warning {{division by zero results in an unspecified value}} \
                             undefined-warning {{division by zero is undefined}}
__constant int m = (-2147483647 - 1) / -1;

int g(int x) {
  switch (x) {
  case 1 / 0: // c-error {{expression is not an integer constant expression}} \
                 cxx-error {{case value is not a constant expression}} \
                 expected-note {{division by zero}}
    return 1;
  }
  return 0;
}
