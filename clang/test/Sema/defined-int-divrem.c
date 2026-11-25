// RUN: %clang_cc1 -fsyntax-only -verify=expected,undefined,c %s
// RUN: %clang_cc1 -fsyntax-only -fdefined-int-divrem -verify=expected,defined,c %s
// RUN: %clang_cc1 -fsyntax-only -ffixed-point -DFIXED -fdefined-int-divrem -verify=expected,defined,c,fixed %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -fdefined-int-divrem -verify=expected,defined,cxx %s

// -Wdivision-by-zero follows -fdefined-int-divrem.

void f(int x) {
  (void)(x / 0); // defined-warning {{division by zero results in an unspecified value}} \
                    undefined-warning {{division by zero is undefined}}
  (void)(x % 0); // defined-warning {{remainder by zero results in an unspecified value}} \
                    undefined-warning {{remainder by zero is undefined}}
}

#ifdef FIXED
// Fixed-point division is not covered by -fdefined-int-divrem.
void fx(_Accum x) {
  (void)(x / 0); // fixed-warning {{division by zero is undefined}}
  (void)(1 / 0.0r);
}
#endif

// Constant evaluation is not changed.
int g(int x) {
  switch (x) {
  case 1 / 0: // c-error {{expression is not an integer constant expression}} \
                 cxx-error {{case value is not a constant expression}} \
                 expected-note {{division by zero}}
    return 1;
  }
  return 0;
}
