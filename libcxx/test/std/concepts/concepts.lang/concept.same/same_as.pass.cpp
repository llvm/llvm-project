//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, class U>
// concept same_as;

#include <concepts>
#include <type_traits>

struct S1 {};
struct S2 {
  int i;

  int& f();
  double g(int x) const;
};
struct S3 {
  int& r;
};
struct S4 {
  int&& r;
};
struct S5 {
  int* p;
};

class C1 {};
class C2 {
  [[maybe_unused]] int i;
};

class C3 {
public:
  int i;
};

template <class T1, class T2 = T1>
class C4 {
  int t1;
  int t2;
};

template <class T1, class T2 = T1>
class C5 {
  [[maybe_unused]] T1 t1;

public:
  T2 t2;
};

template <class T1, class T2 = T1>
class C6 {
public:
  [[maybe_unused]] T1 t1;
  [[maybe_unused]] T2 t2;
};

template <class T>
struct identity {
  using type = T;
};

template <template <typename> class Modifier = identity>
void CheckSameAs() {
  static_assert(
      std::same_as<typename Modifier<int>::type, typename Modifier<int>::type>);
  static_assert(
      std::same_as<typename Modifier<S1>::type, typename Modifier<S1>::type>);
  static_assert(
      std::same_as<typename Modifier<S2>::type, typename Modifier<S2>::type>);
  static_assert(
      std::same_as<typename Modifier<S3>::type, typename Modifier<S3>::type>);
  static_assert(
      std::same_as<typename Modifier<S4>::type, typename Modifier<S4>::type>);
  static_assert(
      std::same_as<typename Modifier<S5>::type, typename Modifier<S5>::type>);
  static_assert(
      std::same_as<typename Modifier<C1>::type, typename Modifier<C1>::type>);
  static_assert(
      std::same_as<typename Modifier<C2>::type, typename Modifier<C2>::type>);
  static_assert(
      std::same_as<typename Modifier<C3>::type, typename Modifier<C3>::type>);
  static_assert(std::same_as<typename Modifier<C4<int> >::type,
                             typename Modifier<C4<int> >::type>);
  static_assert(std::same_as<typename Modifier<C4<int&> >::type,
                             typename Modifier<C4<int&> >::type>);
  static_assert(std::same_as<typename Modifier<C4<int&&> >::type,
                             typename Modifier<C4<int&&> >::type>);
  static_assert(std::same_as<typename Modifier<C5<int> >::type,
                             typename Modifier<C5<int> >::type>);
  static_assert(std::same_as<typename Modifier<C5<int&> >::type,
                             typename Modifier<C5<int&> >::type>);
  static_assert(std::same_as<typename Modifier<C5<int&&> >::type,
                             typename Modifier<C5<int&&> >::type>);
  static_assert(std::same_as<typename Modifier<C6<int> >::type,
                             typename Modifier<C6<int> >::type>);
  static_assert(std::same_as<typename Modifier<C6<int&> >::type,
                             typename Modifier<C6<int&> >::type>);
  static_assert(std::same_as<typename Modifier<C6<int&&> >::type,
                             typename Modifier<C6<int&&> >::type>);

  static_assert(std::same_as<typename Modifier<void>::type,
                             typename Modifier<void>::type>);
}

template <template <typename> class Modifier1,
          template <typename> class Modifier2>
void CheckNotSameAs() {
  static_assert(!std::same_as<typename Modifier1<int>::type,
                              typename Modifier2<int>::type>);
  static_assert(!std::same_as<typename Modifier1<S1>::type,
                              typename Modifier2<S1>::type>);
  static_assert(!std::same_as<typename Modifier1<S2>::type,
                              typename Modifier2<S2>::type>);
  static_assert(!std::same_as<typename Modifier1<S3>::type,
                              typename Modifier2<S3>::type>);
  static_assert(!std::same_as<typename Modifier1<S4>::type,
                              typename Modifier2<S4>::type>);
  static_assert(!std::same_as<typename Modifier1<S5>::type,
                              typename Modifier2<S5>::type>);
  static_assert(!std::same_as<typename Modifier1<C1>::type,
                              typename Modifier2<C1>::type>);
  static_assert(!std::same_as<typename Modifier1<C2>::type,
                              typename Modifier2<C2>::type>);
  static_assert(!std::same_as<typename Modifier1<C3>::type,
                              typename Modifier2<C3>::type>);
  static_assert(!std::same_as<typename Modifier1<C4<int> >::type,
                              typename Modifier2<C4<int> >::type>);
  static_assert(!std::same_as<typename Modifier1<C4<int&> >::type,
                              typename Modifier2<C4<int&> >::type>);
  static_assert(!std::same_as<typename Modifier1<C4<int&&> >::type,
                              typename Modifier2<C4<int&&> >::type>);
  static_assert(!std::same_as<typename Modifier1<C5<int> >::type,
                              typename Modifier2<C5<int> >::type>);
  static_assert(!std::same_as<typename Modifier1<C5<int&> >::type,
                              typename Modifier2<C5<int&> >::type>);
  static_assert(!std::same_as<typename Modifier1<C5<int&&> >::type,
                              typename Modifier2<C5<int&&> >::type>);
  static_assert(!std::same_as<typename Modifier1<C6<int> >::type,
                              typename Modifier2<C6<int> >::type>);
  static_assert(!std::same_as<typename Modifier1<C6<int&> >::type,
                              typename Modifier2<C6<int&> >::type>);
  static_assert(!std::same_as<typename Modifier1<C6<int&&> >::type,
                              typename Modifier2<C6<int&&> >::type>);
}

// Checks subsumption works as intended
template <class T, class U>
requires std::same_as<T, U> void SubsumptionTest();

// clang-format off
template <class T, class U>
requires std::same_as<U, T> && true // NOLINT(readability-simplify-boolean-expr)
int SubsumptionTest();
// clang-format on

static_assert(std::same_as<int, decltype(SubsumptionTest<int, int>())>);
static_assert(std::same_as<int, decltype(SubsumptionTest<void, void>())>);
static_assert(
    std::same_as<int, decltype(SubsumptionTest<int (*)(), int (*)()>())>);
static_assert(
    std::same_as<
        int, decltype(SubsumptionTest<double (&)(int), double (&)(int)>())>);
static_assert(
    std::same_as<int, decltype(SubsumptionTest<int S2::*, int S2::*>())>);
static_assert(
    std::same_as<int,
                 decltype(SubsumptionTest<int& (S2::*)(), int& (S2::*)()>())>);

int main(int, char**) {
  { // Checks std::same_as<T, T> is true
    CheckSameAs();

    // Checks std::same_as<T&, T&> is true
    CheckSameAs<std::add_lvalue_reference>();

    // Checks std::same_as<T&&, T&&> is true
    CheckSameAs<std::add_rvalue_reference>();

    // Checks std::same_as<const T, const T> is true
    CheckSameAs<std::add_const>();

    // Checks std::same_as<volatile T, volatile T> is true
    CheckSameAs<std::add_volatile>();

    // Checks std::same_as<const volatile T, const volatile T> is true
    CheckSameAs<std::add_cv>();

    // Checks std::same_as<T*, T*> is true
    CheckSameAs<std::add_pointer>();

    // Checks concrete types are identical
    static_assert(std::same_as<void, void>);

    using Void = void;
    static_assert(std::same_as<void, Void>);

    static_assert(std::same_as<int[1], int[1]>);
    static_assert(std::same_as<int[2], int[2]>);

    static_assert(std::same_as<int (*)(), int (*)()>);
    static_assert(std::same_as<void (&)(), void (&)()>);
    static_assert(std::same_as<S1& (*)(S1), S1& (*)(S1)>);
    static_assert(std::same_as<C1& (&)(S1, int), C1& (&)(S1, int)>);

    static_assert(std::same_as<int S2::*, int S2::*>);
    static_assert(std::same_as<double S2::*, double S2::*>);

    static_assert(std::same_as<int& (S2::*)(), int& (S2::*)()>);
    static_assert(std::same_as<double& (S2::*)(int), double& (S2::*)(int)>);
  }

  { // Checks that `T` and `T&` are distinct types
    CheckNotSameAs<identity, std::add_lvalue_reference>();
    CheckNotSameAs<std::add_lvalue_reference, identity>();

    // Checks that `T` and `T&&` are distinct types
    CheckNotSameAs<identity, std::add_rvalue_reference>();
    CheckNotSameAs<std::add_rvalue_reference, identity>();

    // Checks that `T` and `const T` are distinct types
    CheckNotSameAs<identity, std::add_const>();
    CheckNotSameAs<std::add_const, identity>();

    // Checks that `T` and `volatile T` are distinct types
    CheckNotSameAs<identity, std::add_volatile>();
    CheckNotSameAs<std::add_volatile, identity>();

    // Checks that `T` and `const volatile T` are distinct types
    CheckNotSameAs<identity, std::add_cv>();
    CheckNotSameAs<std::add_cv, identity>();

    // Checks that `const T` and `volatile T` are distinct types
    CheckNotSameAs<std::add_const, std::add_volatile>();
    CheckNotSameAs<std::add_volatile, std::add_const>();

    // Checks that `const T` and `const volatile T` are distinct types
    CheckNotSameAs<std::add_const, std::add_cv>();
    CheckNotSameAs<std::add_cv, std::add_const>();

    // Checks that `volatile T` and `const volatile T` are distinct types
    CheckNotSameAs<std::add_volatile, std::add_cv>();
    CheckNotSameAs<std::add_cv, std::add_volatile>();

    // Checks `T&` and `T&&` are distinct types
    CheckNotSameAs<std::add_lvalue_reference, std::add_rvalue_reference>();
    CheckNotSameAs<std::add_rvalue_reference, std::add_lvalue_reference>();
  }

  { // Checks different type names are distinct types
    static_assert(!std::same_as<S1, C1>);
    static_assert(!std::same_as<C4<int>, C5<int> >);
    static_assert(!std::same_as<C4<int>, C5<int> >);
    static_assert(!std::same_as<C5<int, double>, C5<double, int> >);

    static_assert(!std::same_as<int&, const int&>);
    static_assert(!std::same_as<int&, volatile int&>);
    static_assert(!std::same_as<int&, const volatile int&>);

    static_assert(!std::same_as<int&&, const int&>);
    static_assert(!std::same_as<int&&, volatile int&>);
    static_assert(!std::same_as<int&&, const volatile int&>);

    static_assert(!std::same_as<int&, const int&&>);
    static_assert(!std::same_as<int&, volatile int&&>);
    static_assert(!std::same_as<int&, const volatile int&&>);

    static_assert(!std::same_as<int&&, const int&&>);
    static_assert(!std::same_as<int&&, volatile int&&>);
    static_assert(!std::same_as<int&&, const volatile int&&>);

    static_assert(!std::same_as<void, int>);

    static_assert(!std::same_as<int[1], int[2]>);
    static_assert(!std::same_as<double[1], int[2]>);

    static_assert(!std::same_as<int* (*)(), const int* (*)()>);
    static_assert(!std::same_as<void (&)(), void (&)(S1)>);
    static_assert(!std::same_as<S1 (*)(S1), S1& (*)(S1)>);
    static_assert(!std::same_as<C3 (&)(int), C1& (&)(S1, int)>);

    static_assert(!std::same_as<int S2::*, double S2::*>);

    static_assert(!std::same_as<int& (S2::*)(), double& (S2::*)(int)>);
  }

  return 0;
}
