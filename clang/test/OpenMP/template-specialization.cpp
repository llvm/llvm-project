// RUN: %clang_cc1 -verify -fopenmp -fsyntax-only %s

// expected-no-diagnostics

template <typename T>
struct z {
  static void aj() {
    T f;
#pragma omp target map(f)
    ;
  }
};

template <typename> class ar {};
template <int> struct as {};
template class z<ar<as<4>>>;
