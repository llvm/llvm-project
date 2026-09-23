// RUN: %clang_cc1 -std=c++2d -x c++-header %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -std=c++2d -x c++ /dev/null -include-pch %t.pch

// RUN: %clang_cc1 -std=c++2d -x c++-header %s -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 -std=c++2d -x c++ /dev/null -include-pch %t.pch

template <class T> struct A {
  using type = T;
};
template <class T> struct B {};
template <template <class> class> struct Take {};

template <unsigned I, template <class> class... TT>
using Indexed = TT...[I]<int>;

template <unsigned I, template <class> class... TT>
using Nested = typename TT...[I]<int>::type;

template <unsigned I, template <class> class... TT>
using AsArgument = Take<TT...[I]>;

template <template <class> class... TT>
struct Base : TT...[0]<int> {};

template <class T> struct Deduce {
  Deduce(T);
};
template <template <class> class... TT>
auto ctad() {
  TT...[0] x{0};
  return x;
}

template <class T> concept Always = true;
template <class T> concept Never = false;
template <class T> constexpr int Var = 1;
template <class T> constexpr int Var2 = 2;

template <unsigned I, template <class> concept... CC>
constexpr bool ConceptId = CC...[I]<int>;

template <unsigned I, template <class> auto... VV>
constexpr int VariableTemplateId = VV...[I]<int>;

template <template <class> concept... CC>
struct Constrained {
  template <CC...[0] T>
  static constexpr int f() { return 3; }
  static constexpr int g(CC...[0] auto) { return 4; }
};

template <template <class> concept... CC>
constexpr int Requires() requires CC...[0]<int> { return 5; }

void fn() {
  static_assert(__is_same(Indexed<1, A, B>, B<int>));
  static_assert(__is_same(Nested<0, A, B>, int));
  static_assert(__is_same(AsArgument<1, A, B>, Take<B>));
  static_assert(__is_base_of(A<int>, Base<A, B>));
  static_assert(__is_same(decltype(ctad<Deduce>()), Deduce<int>));
  static_assert(ConceptId<0, Always, Never>);
  static_assert(!ConceptId<1, Always, Never>);
  static_assert(VariableTemplateId<1, Var, Var2> == 2);
  static_assert(Constrained<Always>::f<int>() == 3);
  static_assert(Constrained<Always>::g(0) == 4);
  static_assert(Requires<Always>() == 5);
}
