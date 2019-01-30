// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=c++
// expected-no-diagnostics

struct RetGlob {
  int dummy;
};

struct RetGen {
  char dummy;
};

RetGlob foo(const __global int *);
RetGen foo(const __generic int *);

void kernel k() {
  __global int *ArgGlob;
  __generic int *ArgGen;
  __local int *ArgLoc;
  RetGlob TestGlob = foo(ArgGlob);
  RetGen TestGen = foo(ArgGen);
  TestGen = foo(ArgLoc);
}
