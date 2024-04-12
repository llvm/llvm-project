// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s

// ref-no-diagnostics

typedef int __attribute__((vector_size(16))) VI4;
constexpr VI4 A = {1,2,3,4};

/// From constant-expression-cxx11.cpp
namespace Vector {
  typedef int __attribute__((vector_size(16))) VI4;
  constexpr VI4 f(int n) {
    return VI4 { n * 3, n + 4, n - 5, n / 6 };
  }
  constexpr auto v1 = f(10);

  typedef double __attribute__((vector_size(32))) VD4;
  constexpr VD4 g(int n) {
    return (VD4) { n / 2.0, n + 1.5, n - 5.4, n * 0.9 };
  }
  constexpr auto v2 = g(4);
}

/// FIXME: We need to support BitCasts between vector types.
namespace {
  typedef float __attribute__((vector_size(16))) VI42;
  constexpr VI42 A2 = A; // expected-error {{must be initialized by a constant expression}}
}
