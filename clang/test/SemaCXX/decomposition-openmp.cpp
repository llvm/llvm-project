// RUN: %clang_cc1 -fsyntax-only -verify=cxx17 -std=c++17 -fopenmp %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -fopenmp %s

// expected-no-diagnostics

// Okay, not an OpenMP capture.
auto f() {
  int i[2] = {};
  // cxx17-note@+2{{'a' declared here}}
  // cxx17-note@+1{{'b' declared here}}
  auto [a, b] = i;
  // cxx17-warning@+1{{captured structured bindings are a C++20 extension}}
  return [=, &a] {
    // cxx17-warning@+1{{captured structured bindings are a C++20 extension}}
    return a + b;
  };
}

// Okay, not an OpenMP capture.
void foo(int);
void g() {
  #pragma omp parallel
  {
    int i[2] = {};
    // cxx17-note@+2{{'a' declared here}}
    // cxx17-note@+1{{'b' declared here}}
    auto [a, b] = i;
    // cxx17-warning@+2{{captured structured bindings are a C++20 extension}}
    // cxx17-warning@+1{{captured structured bindings are a C++20 extension}}
    auto L = [&] { foo(a+b); };
  }
}

void h() {
  int i[2] = {};
  // cxx17-note@+2{{'a' declared here}}
  // cxx17-note@+1{{'b' declared here}}
  auto [a, b] = i;
  #pragma omp parallel
  {
    // cxx17-warning@+2{{captured structured bindings are a C++20 extension}}
    // cxx17-warning@+1{{captured structured bindings are a C++20 extension}}
    foo(a + b);
  }
}
