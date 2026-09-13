// RUN: %clang_cc1 -fms-extensions -DMS_EXTENSIONS -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fms-extensions -DMS_EXTENSIONS -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcuda-is-device -fms-extensions -DMS_EXTENSIONS -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fcuda-is-device -fms-extensions -DMS_EXTENSIONS -fsyntax-only -verify %s
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
// list, body, template parameter list, or lambda declarator ending in an
// unambiguous continuation. A Microsoft attribute is followed by the
// declaration it appertains to.

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

void interleaved_noinline() {
  Wrapper a([] __device__ __noinline__ () {});
  Wrapper b([] __attribute__((device)) __noinline__
                __attribute__((host)) () {});
  Wrapper c([] __device__ __noinline__ {});
  (void)a; (void)b; (void)c;
}

#if __cplusplus >= 202302L
void omitted_parameter_list() {
  Wrapper a([] __device__ noexcept {});
  Wrapper b([] __device__ -> int { return 1; });
  Wrapper c([] __device__ mutable {});
  Wrapper d([] __device__ constexpr {});
  Wrapper e([] __device__ consteval {});
  Wrapper f([] __device__ static {});
  Wrapper g([] __device__ mutable constexpr noexcept -> int { return 1; });
  Wrapper h([] __device__ mutable [[]] {});
  Wrapper i([] __device__ __noinline__ mutable constexpr noexcept -> int {
    return 1;
  });
  (void)a; (void)b; (void)c; (void)d;
  (void)e; (void)f; (void)g; (void)h; (void)i;
}

void explicit_template_parameter_list() {
  Wrapper a([] __device__ <typename T>(T x) { return x; });
  (void)a;
}
#endif

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
#if __cplusplus >= 202302L
  [] mutable int member;
  [] static constexpr int constant = 1;
  [] consteval int immediate();
#endif
};
#endif
