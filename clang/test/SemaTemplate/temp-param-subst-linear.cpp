// RUN: %clang_cc1 -std=c++17 %s -verify
// expected-no-diagnostics

// This test attempts to ensure that the below template parameter pack
// splitting technique executes in linear time in the number of template
// parameters. The size of the list below is selected so as to execute
// relatively quickly on a "good" compiler and to time out otherwise.

template<typename...> struct TypeList;

namespace detail {
template<unsigned> using Unsigned = unsigned;
template<typename T, T ...N> using ListOfNUnsignedsImpl = TypeList<Unsigned<N>...>;
template<unsigned N> using ListOfNUnsigneds =
  __make_integer_seq<ListOfNUnsignedsImpl, unsigned, N>;

template<typename T> struct TypeWrapper {
  template<unsigned> using AsTemplate = T;
};

template<typename ...N> struct Splitter {
  template<template<N> class ...L,
           template<unsigned> class ...R> struct Split {
    using Left = TypeList<L<0>...>;
    using Right = TypeList<R<0>...>;
  };
};
}

template<typename TypeList, unsigned N, typename = detail::ListOfNUnsigneds<N>>
struct SplitAtIndex;

template<typename ...T, unsigned N, typename ...NUnsigneds>
struct SplitAtIndex<TypeList<T...>, N, TypeList<NUnsigneds...>> :
  detail::Splitter<NUnsigneds...>::
    template Split<detail::TypeWrapper<T>::template AsTemplate...> {};

template<typename T, int N> struct Rep : Rep<typename Rep<T, N-1>::type, 1> {};
template<typename ...T> struct Rep<TypeList<T...>, 1> { typedef TypeList<T..., T...> type; };

using Ints = Rep<TypeList<int>, 14>::type;

template<typename T> extern int Size;
template<typename ...T> constexpr int Size<TypeList<T...>> = sizeof...(T);

using Left = SplitAtIndex<Ints, Size<Ints> / 2>::Left;
using Right = SplitAtIndex<Ints, Size<Ints> / 2>::Right;
static_assert(Size<Left> == 8192);
static_assert(Size<Right> == 8192);

template<typename L, typename R> struct Concat;
template<typename ...L, typename ...R> struct Concat<TypeList<L...>, TypeList<R...>> {
  using type = TypeList<L..., R...>;
};

using Ints = Concat<Left, Right>::type;
