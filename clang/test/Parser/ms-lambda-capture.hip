// RUN: %clang_cc1 -fms-extensions -DMS_EXTENSIONS -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcuda-is-device -fms-extensions -DMS_EXTENSIONS -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify %s
// expected-no-diagnostics

// A '[' can begin a Microsoft attribute list under -fms-extensions, but it can
// also begin a C++ lambda-introducer. A lambda passed to a constructor via CTAD
// must not have its capture list consumed as a (possibly empty) Microsoft
// attribute, which previously misparsed the initializer as a function
// declarator under -fms-extensions.
//
// A lambda and a Microsoft attribute are told apart by what follows the
// (possibly attributed) '[...]' introducer. A lambda continues with a parameter
// list or body, whereas a Microsoft attribute is followed by the declaration
// it appertains to.

#include "Inputs/cuda.h"

template <typename F> struct Wrapper {
  F f;
  __host__ Wrapper(F fn) : f(fn) {}
};
template <typename F> Wrapper(F) -> Wrapper<F>;

void empty_and_defaults() {
  Wrapper a([] __device__ () {});
  Wrapper b([=] __device__ () {});
  Wrapper c([&] __device__ () {});
  Wrapper d([] __device__ {});
  (void)a; (void)b; (void)c; (void)d;
}

void named_captures() {
  int values[] = {1};
  int x = 0;
  Wrapper a([x] __device__ () { return x; });
  Wrapper b([&x] __device__ () { x++; });
  Wrapper c([y = x] __device__ () { return y; });
  Wrapper d([x, &x2 = x] __device__ () { return x + x2; });
  Wrapper e([a = values[0], b = [] { return 2; }()] __device__ () {
    return a + b;
  });
  (void)a; (void)b; (void)c; (void)d; (void)e;
}

struct S {
  int m = 0;
  void g() {
    Wrapper a([this] __device__ () { return m; });
    Wrapper b([*this] __device__ () { return m; });
    (void)a; (void)b;
  }
};

#ifdef MS_EXTENSIONS
struct MicrosoftAttributeControls {
  [] static int empty;
  [propget] int get();
};
#endif
