// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fexperimental-new-constant-interpreter %s

constexpr unsigned long long A(unsigned long long m, unsigned long long n) {
  return m == 0 ? n + 1 : n == 0 ? A(m-1, 1) : A(m - 1, A(m, n - 1));
}

using X = int[A(3,4)];
using X = int[125];
