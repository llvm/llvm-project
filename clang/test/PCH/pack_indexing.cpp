// RUN: %clang_cc1 -std=c++2c -x c++-header %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -std=c++2c -x c++ /dev/null -include-pch %t.pch

// RUN: %clang_cc1 -std=c++2c -x c++-header %s -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 -std=c++2c -x c++ /dev/null -include-pch %t.pch

template <int I, typename... U>
using Type = U...[I];

template <int I, auto...V>
constexpr auto Var = V...[I];

void fn1() {
  using A = Type<1, int, long, double>;
  constexpr auto V = Var<2, 0, 1, 42>;
}
