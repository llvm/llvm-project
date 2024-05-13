//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// common_type

#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct E {};

template <class T>
struct X { explicit X(T const&){} };

template <class T>
struct S { explicit S(T const&){} };

template <class T>
struct bad_reference_wrapper {
    bad_reference_wrapper(T&);
    bad_reference_wrapper(T&&) = delete;
    operator T&() const;
};

namespace std
{
    template <typename T>
    struct common_type<T, ::S<T> >
    {
        typedef S<T> type;
    };

    template <class T>
    struct common_type< ::S<T>, T> {
      typedef S<T> type;
    };

//  P0548
    template <class T>
    struct common_type< ::S<T>, ::S<T> > {
      typedef S<T> type;
    };

    template <> struct common_type< ::S<long>, long> {};
    template <> struct common_type<long, ::S<long> > {};
    template <> struct common_type< ::X<double>, ::X<double> > {};
}

template <class> struct VoidT { typedef void type; };

#if TEST_STD_VER < 11
template <class Trait, class = void>
struct no_common_type_imp : std::true_type {};

template <class Trait>
struct no_common_type_imp<Trait, typename VoidT<typename Trait::type>::type>
    : std::false_type {};

struct NoArgTag;

template <class Tp = NoArgTag, class Up = NoArgTag, class Vp = NoArgTag>
struct no_common_type : no_common_type_imp<std::common_type<Tp, Up, Vp> > {};
template <class Tp, class Up>
struct no_common_type<Tp, Up> : no_common_type_imp<std::common_type<Tp, Up> > {
};
template <class Tp>
struct no_common_type<Tp> : no_common_type_imp<std::common_type<Tp> > {};
template <>
struct no_common_type<> : no_common_type_imp<std::common_type<> > {};
#else
template <class Tp>
struct always_bool_imp { using type = bool; };
template <class Tp> using always_bool = typename always_bool_imp<Tp>::type;

template <class ...Args>
constexpr auto no_common_type_imp(int)
-> always_bool<typename std::common_type<Args...>::type>
{ return false; }

template <class ...Args>
constexpr bool no_common_type_imp(long) { return true; }

template <class ...Args>
using no_common_type = std::integral_constant<bool, no_common_type_imp<Args...>(0)>;
#endif

template <class T1, class T2>
struct TernaryOp {
  static_assert((std::is_same<typename std::decay<T1>::type, T1>::value), "must be same");
  static_assert((std::is_same<typename std::decay<T2>::type, T2>::value), "must be same");
  typedef typename std::decay<
      decltype(false ? std::declval<T1>() : std::declval<T2>())
    >::type type;
};

// (4.1)
// -- If sizeof...(T) is zero, there shall be no member type.
void test_bullet_one() {
  static_assert(no_common_type<>::value, "");
}

// (4.2)
// -- If sizeof...(T) is one, let T0 denote the sole type constituting the pack
//    T. The member typedef-name type shall denote the same type, if any, as
//    common_type_t<T0, T0>; otherwise there shall be no member type.
void test_bullet_two() {
  static_assert((std::is_same<std::common_type<void>::type, void>::value), "");
  static_assert((std::is_same<std::common_type<int>::type, int>::value), "");
  static_assert((std::is_same<std::common_type<int const>::type, int>::value), "");
  static_assert((std::is_same<std::common_type<int volatile[]>::type, int volatile*>::value), "");
  static_assert((std::is_same<std::common_type<void(&)()>::type, void(*)()>::value), "");

  static_assert((no_common_type<X<double> >::value), "");
}

template <class T, class U, class Expect>
void test_bullet_three_one_imp() {
  typedef typename std::decay<T>::type DT;
  typedef typename std::decay<U>::type DU;
  static_assert((!std::is_same<T, DT>::value || !std::is_same<U, DU>::value), "");
  static_assert((std::is_same<typename std::common_type<T, U>::type, Expect>::value), "");
  static_assert((std::is_same<typename std::common_type<U, T>::type, Expect>::value), "");
  static_assert((std::is_same<typename std::common_type<T, U>::type, typename std::common_type<DT, DU>::type>::value), "");
}

// (4.3)
// -- If sizeof...(T) is two, let the first and second types constituting T be
//    denoted by T1 and T2, respectively, and let D1 and D2 denote the same types
//    as decay_t<T1> and decay_t<T2>, respectively.
// (4.3.1)
//    -- If is_same_v<T1, D1> is false or is_same_v<T2, D2> is false, let C
//       denote the same type, if any, as common_type_t<D1, D2>.
void test_bullet_three_one() {
  // Test that the user provided specialization of common_type is used after
  // decaying T1.
  {
    typedef const S<int> T1;
    typedef int T2;
    test_bullet_three_one_imp<T1, T2, S<int> >();
  }
  // Test a user provided specialization that does not provide a typedef.
  {
    typedef const ::S<long> T1;
    typedef long T2;
    static_assert((no_common_type<T1, T2>::value), "");
    static_assert((no_common_type<T2, T1>::value), "");
  }
  // Test that the ternary operator is not applied when the types are the
  // same.
  {
    typedef const void T1;
    typedef void Expect;
    static_assert((std::is_same<std::common_type<T1, T1>::type, Expect>::value), "");
    static_assert((std::is_same<std::common_type<T1, T1>::type, std::common_type<T1>::type>::value), "");
  }
  {
    typedef int const T1[];
    typedef int const* Expect;
    static_assert((std::is_same<std::common_type<T1, T1>::type, Expect>::value), "");
    static_assert((std::is_same<std::common_type<T1, T1>::type, std::common_type<T1>::type>::value), "");
  }
}

// (4.3)
// -- If sizeof...(T) is two, let the first and second types constituting T be
//    denoted by T1 and T2, respectively, and let D1 and D2 denote the same types
//    as decay_t<T1> and decay_t<T2>, respectively.
// (4.3.1)
//    -- If [...]
// (4.3.2)
//    -- [Note: [...]
// (4.3.3)
//    -- Otherwise, if
//       decay_t<decltype(false ? declval<D1>() : declval<D2>())>
//       denotes a type, let C denote that type.
void test_bullet_three_three() {
  {
    typedef int const* T1;
    typedef int* T2;
    typedef TernaryOp<T1, T2>::type Expect;
    static_assert((std::is_same<std::common_type<T1, T2>::type, Expect>::value), "");
    static_assert((std::is_same<std::common_type<T2, T1>::type, Expect>::value), "");
  }
  // Test that there is no ::type member when the ternary op is ill-formed
  {
    typedef int T1;
    typedef void T2;
    static_assert((no_common_type<T1, T2>::value), "");
    static_assert((no_common_type<T2, T1>::value), "");
  }
  {
    typedef int T1;
    typedef X<int> T2;
    static_assert((no_common_type<T1, T2>::value), "");
    static_assert((no_common_type<T2, T1>::value), "");
  }
  // Test that the ternary operator is not applied when the types are the
  // same.
  {
    typedef void T1;
    typedef void Expect;
    static_assert((std::is_same<std::common_type<T1, T1>::type, Expect>::value), "");
    static_assert((std::is_same<std::common_type<T1, T1>::type, std::common_type<T1>::type>::value), "");
  }
}

// (4.3)
// -- If sizeof...(T) is two, let the first and second types constituting T be
//    denoted by T1 and T2, respectively, and let D1 and D2 denote the same types
//    as decay_t<T1> and decay_t<T2>, respectively.
// (4.3.1)
//    -- If [...]
// (4.3.2)
//    -- [Note: [...]
// (4.3.3)
//    -- Otherwise
// (4.3.4)
//    -- Otherwise, if COND-RES(CREF(D1), CREF(D2)) denotes a type, let C
//       denote the type decay_t<COND-RES(CREF(D1), CREF(D2))>.
void test_bullet_three_four() {
#if TEST_STD_VER >= 20
  static_assert(std::is_same_v<std::common_type_t<int, bad_reference_wrapper<int>>, int>, "");
  static_assert(std::is_same_v<std::common_type_t<bad_reference_wrapper<double>, double>, double>, "");
  static_assert(std::is_same_v<std::common_type_t<const bad_reference_wrapper<double>, double>, double>, "");
  static_assert(std::is_same_v<std::common_type_t<volatile bad_reference_wrapper<double>, double>, double>, "");
  static_assert(std::is_same_v<std::common_type_t<const volatile bad_reference_wrapper<double>, double>, double>, "");

  static_assert(std::is_same_v<std::common_type_t<bad_reference_wrapper<double>, const double>, double>, "");
  static_assert(std::is_same_v<std::common_type_t<bad_reference_wrapper<double>, volatile double>, double>, "");
  static_assert(std::is_same_v<std::common_type_t<bad_reference_wrapper<double>, const volatile double>, double>, "");

  static_assert(std::is_same_v<std::common_type_t<bad_reference_wrapper<double>&, double>, double>, "");
  static_assert(std::is_same_v<std::common_type_t<bad_reference_wrapper<double>, double&>, double>, "");
#endif
}

// (4.4)
// -- If sizeof...(T) is greater than two, let T1, T2, and R, respectively,
// denote the first, second, and (pack of) remaining types constituting T.
// Let C denote the same type, if any, as common_type_t<T1, T2>. If there is
// such a type C, the member typedef-name type shall denote the
// same type, if any, as common_type_t<C, R...>. Otherwise, there shall be
// no member type.
void test_bullet_four() {
  { // test that there is no ::type member
    static_assert((no_common_type<int, E>::value), "");
    static_assert((no_common_type<int, int, E>::value), "");
#if TEST_STD_VER >= 11
    static_assert((no_common_type<int, int, E, int>::value), "");
    static_assert((no_common_type<int, int, int, E>::value), "");
#endif
  }
}

#if TEST_STD_VER > 20
struct A {};
struct B {};
struct C : B {};
template<>
struct std::common_type<A, std::tuple<B>> {
  using type = tuple<B>;
};
#endif

int main(int, char**)
{
    static_assert((std::is_same<std::common_type<int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<char>::type, char>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<int>,   int>::value), "");
    static_assert((std::is_same<std::common_type_t<char>, char>::value), "");
#endif

    static_assert((std::is_same<std::common_type<               int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<const          int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<      volatile int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<const volatile int>::type, int>::value), "");

    static_assert((std::is_same<std::common_type<int,           int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<int,     const int>::type, int>::value), "");

    static_assert((std::is_same<std::common_type<long,       const int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<const long,       int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<long,    volatile int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<volatile long,    int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<const long, const int>::type, long>::value), "");

    static_assert((std::is_same<std::common_type<double, char>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<short, char>::type, int>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<double, char>, double>::value), "");
    static_assert((std::is_same<std::common_type_t<short, char>, int>::value), "");
#endif

    static_assert((std::is_same<std::common_type<double, char, long long>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<unsigned, char, long long>::type, long long>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<double, char, long long>, double>::value), "");
    static_assert((std::is_same<std::common_type_t<unsigned, char, long long>, long long>::value), "");
#endif

    static_assert((std::is_same<std::common_type<               void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const          void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<      volatile void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const volatile void>::type, void>::value), "");

    static_assert((std::is_same<std::common_type<void,       const void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const void,       void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<void,    volatile void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<volatile void,    void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const void, const void>::type, void>::value), "");

    static_assert((std::is_same<std::common_type<int, S<int> >::type, S<int> >::value), "");
    static_assert((std::is_same<std::common_type<int, S<int>, S<int> >::type, S<int> >::value), "");
    static_assert((std::is_same<std::common_type<int, int, S<int> >::type, S<int> >::value), "");


  test_bullet_one();
  test_bullet_two();
  test_bullet_three_one();
  test_bullet_three_three();
  test_bullet_three_four();
  test_bullet_four();

    // P0548
    static_assert((std::is_same<std::common_type<S<int> >::type,         S<int> >::value), "");
    static_assert((std::is_same<std::common_type<S<int>, S<int> >::type, S<int> >::value), "");

    static_assert((std::is_same<std::common_type<int>::type,                int>::value), "");
    static_assert((std::is_same<std::common_type<const int>::type,          int>::value), "");
    static_assert((std::is_same<std::common_type<volatile int>::type,       int>::value), "");
    static_assert((std::is_same<std::common_type<const volatile int>::type, int>::value), "");

    static_assert((std::is_same<std::common_type<int, int>::type,             int>::value), "");
    static_assert((std::is_same<std::common_type<const int, int>::type,       int>::value), "");
    static_assert((std::is_same<std::common_type<int, const int>::type,       int>::value), "");
    static_assert((std::is_same<std::common_type<const int, const int>::type, int>::value), "");

#if TEST_STD_VER >= 11
    // Test that we're really variadic in C++11
    static_assert(std::is_same<std::common_type<int, int, int, int, int, int, int, int>::type, int>::value, "");
#endif

#if TEST_STD_VER > 20
    // P2321
    static_assert(std::is_same_v<std::common_type_t<std::tuple<int>>, std::tuple<int>>);
    static_assert(std::is_same_v<std::common_type_t<std::tuple<int>, std::tuple<long>>, std::tuple<long>>);
    static_assert(std::is_same_v<std::common_type_t<std::tuple<const int>, std::tuple<const int>>, std::tuple<int>>);
    static_assert(std::is_same_v<std::common_type_t<std::tuple<const int&>>, std::tuple<int>>);
    static_assert(std::is_same_v<std::common_type_t<std::tuple<const volatile int&>, std::tuple<const volatile long&>>, std::tuple<long>>);

    static_assert(std::is_same_v<std::common_type_t<A, std::tuple<B>, std::tuple<C>>, std::tuple<B>>);

    static_assert(std::is_same_v<std::common_type_t<std::pair<int, int>>, std::pair<int, int>>);
    static_assert(std::is_same_v<std::common_type_t<std::pair<int, long>, std::pair<long, int>>, std::pair<long, long>>);
    static_assert(std::is_same_v<std::common_type_t<std::pair<const int, const long>,
                                                    std::pair<const int, const long>>,
                                                    std::pair<int, long>>);

    static_assert(std::is_same_v<std::common_type_t<std::pair<const int&, const long&>>, std::pair<int, long>>);
    static_assert(std::is_same_v<std::common_type_t<std::pair<const volatile int&, const volatile long&>,
                                                    std::pair<const volatile long&, const volatile int&>>,
                                                    std::pair<long, long>>);

    static_assert(std::is_same_v<std::common_type_t<A, std::tuple<B>, std::tuple<C>>, std::tuple<B>>);
#endif

  return 0;
}
