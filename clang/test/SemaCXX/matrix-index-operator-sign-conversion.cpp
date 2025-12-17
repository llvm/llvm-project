// RUN: %clang_cc1 -triple arm64-apple-macosx -std=c++11 -fenable-matrix -fsyntax-only -verify -Wsign-conversion %s

template <typename T, int R, int C> using m __attribute__((__matrix_type__(R,C))) = T;

double index1(m<double,3,1> X, int      i) { return X[i][0]; }

double index2(m<double,3,1> X, unsigned i) { return X[i][0]; }

double index3(m<double,3,1> X, char     i) { return X[i][0]; }

double index4(m<double,3,1> X, int      i) { return X[0][i]; }

double index5(m<double,3,1> X, unsigned i) { return X[0][i]; }

double index6(m<double,3,1> X, char     i) { return X[0][i]; }

// expected-no-diagnostics
