// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s

constexpr unsigned oddfac(unsigned n) {
  return n == 1 ? 1 : n * oddfac(n-2);
}
constexpr unsigned k = oddfac(123);

using A = int[k % 256];
using A = int[43];
