// RUN: %clang_cc1 -std=c++14 -x c++-header %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -std=c++14 -x c++ /dev/null -include-pch %t.pch

template <int i>
struct X { };

using SizeT = decltype(sizeof(int));

template <SizeT i, typename ...T>
using TypePackElement = __type_pack_element<i, T...>;

void fn1() {
  X<0> x0 = TypePackElement<0, X<0>, X<1>, X<2>>{};
  X<1> x1 = TypePackElement<1, X<0>, X<1>, X<2>>{};
  X<2> x2 = TypePackElement<2, X<0>, X<1>, X<2>>{};
}
