// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -fopenmp %s

// expected-no-diagnostics

// Okay, not an OpenMP capture.
auto f() {
  int i[2] = {};
  auto [a, b] = i;
  return [=, &a] {
    return a + b;
  };
}

// Okay, not an OpenMP capture.
void foo(int);
void g() {
  #pragma omp parallel
  {
    int i[2] = {};
    auto [a, b] = i;
    auto L = [&] { foo(a+b); };
  }
}

// FIXME: OpenMP should support capturing structured bindings
void h() {
  int i[2] = {};
  auto [a, b] = i;
  #pragma omp parallel
  {
    foo(a + b);
  }
}
