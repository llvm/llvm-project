// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-linux-gnu -ferror-limit 100 %s
// expected-no-diagnostics

// Multiplication reduction over a class type that *is* constructible from the
// integer literal '1' (e.g. std::complex-like) is accepted: the multiplicative
// identity is built via the converting constructor. See
// for_reduction_class_identity_codegen.cpp for the value check.
struct WithInt {
  double V;
  WithInt(int X = 1) : V(X) {}
  WithInt &operator*=(const WithInt &RHS) {
    V *= RHS.V;
    return *this;
  }
};

void ok(WithInt *Data) {
  WithInt Product;
#pragma omp parallel for reduction(* : Product)
  for (int I = 0; I < 4; ++I)
    Product *= Data[I];
}

// Multiplication reduction over a class type that is NOT constructible from '1'
// must NOT be rejected: we fall back to value-initialization (the pre-existing
// behavior) instead of emitting a hard error, so code that used to compile keeps
// compiling.
struct NoInt {
  double V;
  NoInt &operator*=(const NoInt &RHS) {
    V *= RHS.V;
    return *this;
  }
};

void fallback(NoInt *Data) {
  NoInt Product;
#pragma omp parallel for reduction(* : Product)
  for (int I = 0; I < 4; ++I)
    Product *= Data[I];
}
