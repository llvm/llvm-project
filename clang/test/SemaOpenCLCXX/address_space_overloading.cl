// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=c++

// FIXME: This test shouldn't trigger any errors.

struct RetGlob {
  int dummy;
};

struct RetGen { //expected-error{{binding value of type '__generic RetGen' to reference to type 'RetGen' drops <<ERROR>> qualifiers}}
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
  TestGen = foo(ArgLoc); //expected-note{{in implicit copy assignment operator for 'RetGen' first required here}}
}
