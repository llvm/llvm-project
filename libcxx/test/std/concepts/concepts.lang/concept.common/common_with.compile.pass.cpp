//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class From, class To>
// concept common_with;

#include <concepts>
#include <type_traits>

#include "test_macros.h"

template <class T, class U>
constexpr bool CheckCommonWith() noexcept {
  constexpr bool result = std::common_with<T, U>;
  static_assert(std::common_with<T, U&> == result);
  static_assert(std::common_with<T, const U&> == result);
  static_assert(std::common_with<T, volatile U&> == result);
  static_assert(std::common_with<T, const volatile U&> == result);
  static_assert(std::common_with<T, U&&> == result);
  static_assert(std::common_with<T, const U&&> == result);
  static_assert(std::common_with<T, volatile U&&> == result);
  static_assert(std::common_with<T, const volatile U&&> == result);
  static_assert(std::common_with<T&, U&&> == result);
  static_assert(std::common_with<T&, const U&&> == result);
  static_assert(std::common_with<T&, volatile U&&> == result);
  static_assert(std::common_with<T&, const volatile U&&> == result);
  static_assert(std::common_with<const T&, U&&> == result);
  static_assert(std::common_with<const T&, const U&&> == result);
  static_assert(std::common_with<const T&, volatile U&&> == result);
  static_assert(std::common_with<const T&, const volatile U&&> == result);
  static_assert(std::common_with<volatile T&, U&&> == result);
  static_assert(std::common_with<volatile T&, const U&&> == result);
  static_assert(std::common_with<volatile T&, volatile U&&> == result);
  static_assert(std::common_with<volatile T&, const volatile U&&> == result);
  static_assert(std::common_with<const volatile T&, U&&> == result);
  static_assert(std::common_with<const volatile T&, const U&&> == result);
  static_assert(std::common_with<const volatile T&, volatile U&&> == result);
  static_assert(std::common_with<const volatile T&, const volatile U&&> ==
                result);
  return result;
}

template <class T, class U>
constexpr bool HasValidCommonType() noexcept {
  return requires { typename std::common_type_t<T, U>; }
  &&std::same_as<std::common_type_t<T, U>, std::common_type_t<U, T> >;
}

namespace BuiltinTypes {
// fundamental types
static_assert(std::common_with<void, void>);
static_assert(CheckCommonWith<int, int>());
static_assert(CheckCommonWith<int, long>());
static_assert(CheckCommonWith<int, unsigned char>());
#ifndef TEST_HAS_NO_INT128
static_assert(CheckCommonWith<int, __int128_t>());
#endif
static_assert(CheckCommonWith<int, double>());

// arrays
static_assert(CheckCommonWith<int[5], int[5]>());

// pointers
static_assert(CheckCommonWith<int*, int*>());
static_assert(CheckCommonWith<int*, const int*>());
static_assert(CheckCommonWith<int*, volatile int*>());
static_assert(CheckCommonWith<int*, const volatile int*>());
static_assert(CheckCommonWith<const int*, const int*>());
static_assert(CheckCommonWith<const int*, volatile int*>());
static_assert(CheckCommonWith<const int*, const volatile int*>());
static_assert(CheckCommonWith<volatile int*, const int*>());
static_assert(CheckCommonWith<volatile int*, volatile int*>());
static_assert(CheckCommonWith<volatile int*, const volatile int*>());
static_assert(CheckCommonWith<const volatile int*, const int*>());
static_assert(CheckCommonWith<const volatile int*, volatile int*>());
static_assert(CheckCommonWith<const volatile int*, const volatile int*>());

static_assert(CheckCommonWith<int (*)(), int (*)()>());
static_assert(CheckCommonWith<int (*)(), int (*)() noexcept>());
static_assert(CheckCommonWith<int (&)(), int (&)()>());
static_assert(CheckCommonWith<int (&)(), int (&)() noexcept>());
static_assert(CheckCommonWith<int (&)(), int (*)()>());
static_assert(CheckCommonWith<int (&)(), int (*)() noexcept>());

struct S {};
static_assert(CheckCommonWith<int S::*, int S::*>());
static_assert(CheckCommonWith<int S::*, const int S::*>());
static_assert(CheckCommonWith<int (S::*)(), int (S::*)()>());
static_assert(CheckCommonWith<int (S::*)(), int (S::*)() noexcept>());
static_assert(CheckCommonWith<int (S::*)() const, int (S::*)() const>());
static_assert(
    CheckCommonWith<int (S::*)() const, int (S::*)() const noexcept>());
static_assert(CheckCommonWith<int (S::*)() volatile, int (S::*)() volatile>());
static_assert(
    CheckCommonWith<int (S::*)() volatile, int (S::*)() volatile noexcept>());
static_assert(CheckCommonWith<int (S::*)() const volatile,
                              int (S::*)() const volatile>());
static_assert(CheckCommonWith<int (S::*)() const volatile,
                              int (S::*)() const volatile noexcept>());

// nonsense
static_assert(!CheckCommonWith<double, float*>());
static_assert(!CheckCommonWith<int, int[5]>());
static_assert(!CheckCommonWith<int*, long*>());
static_assert(!CheckCommonWith<int*, unsigned int*>());
static_assert(!CheckCommonWith<int (*)(), int (*)(int)>());
static_assert(!CheckCommonWith<int S::*, float S::*>());
static_assert(!CheckCommonWith<int (S::*)(), int (S::*)() const>());
static_assert(!CheckCommonWith<int (S::*)(), int (S::*)() volatile>());
static_assert(!CheckCommonWith<int (S::*)(), int (S::*)() const volatile>());
static_assert(!CheckCommonWith<int (S::*)() const, int (S::*)() volatile>());
static_assert(
    !CheckCommonWith<int (S::*)() const, int (S::*)() const volatile>());
static_assert(
    !CheckCommonWith<int (S::*)() volatile, int (S::*)() const volatile>());
} // namespace BuiltinTypes

namespace NoDefaultCommonType {
class T {};

static_assert(!CheckCommonWith<T, int>());
static_assert(!CheckCommonWith<int, T>());
static_assert(!CheckCommonWith<T, int[10]>());
static_assert(!CheckCommonWith<T[10], int>());
static_assert(!CheckCommonWith<T*, int*>());
static_assert(!CheckCommonWith<T*, const int*>());
static_assert(!CheckCommonWith<T*, volatile int*>());
static_assert(!CheckCommonWith<T*, const volatile int*>());
static_assert(!CheckCommonWith<const T*, int*>());
static_assert(!CheckCommonWith<volatile T*, int*>());
static_assert(!CheckCommonWith<const volatile T*, int*>());
static_assert(!CheckCommonWith<const T*, const int*>());
static_assert(!CheckCommonWith<const T*, volatile int*>());
static_assert(!CheckCommonWith<const T*, const volatile int*>());
static_assert(!CheckCommonWith<const T*, const int*>());
static_assert(!CheckCommonWith<volatile T*, const int*>());
static_assert(!CheckCommonWith<const volatile T*, const int*>());
static_assert(!CheckCommonWith<volatile T*, const int*>());
static_assert(!CheckCommonWith<volatile T*, volatile int*>());
static_assert(!CheckCommonWith<volatile T*, const volatile int*>());
static_assert(!CheckCommonWith<const T*, volatile int*>());
static_assert(!CheckCommonWith<volatile T*, volatile int*>());
static_assert(!CheckCommonWith<const volatile T*, volatile int*>());
static_assert(!CheckCommonWith<const volatile T*, const int*>());
static_assert(!CheckCommonWith<const volatile T*, volatile int*>());
static_assert(!CheckCommonWith<const volatile T*, const volatile int*>());
static_assert(!CheckCommonWith<const T*, const volatile int*>());
static_assert(!CheckCommonWith<volatile T*, const volatile int*>());
static_assert(!CheckCommonWith<const volatile T*, const volatile int*>());
static_assert(!CheckCommonWith<T&, int&>());
static_assert(!CheckCommonWith<T&, const int&>());
static_assert(!CheckCommonWith<T&, volatile int&>());
static_assert(!CheckCommonWith<T&, const volatile int&>());
static_assert(!CheckCommonWith<const T&, int&>());
static_assert(!CheckCommonWith<volatile T&, int&>());
static_assert(!CheckCommonWith<const volatile T&, int&>());
static_assert(!CheckCommonWith<const T&, const int&>());
static_assert(!CheckCommonWith<const T&, volatile int&>());
static_assert(!CheckCommonWith<const T&, const volatile int&>());
static_assert(!CheckCommonWith<const T&, const int&>());
static_assert(!CheckCommonWith<volatile T&, const int&>());
static_assert(!CheckCommonWith<const volatile T&, const int&>());
static_assert(!CheckCommonWith<volatile T&, const int&>());
static_assert(!CheckCommonWith<volatile T&, volatile int&>());
static_assert(!CheckCommonWith<volatile T&, const volatile int&>());
static_assert(!CheckCommonWith<const T&, volatile int&>());
static_assert(!CheckCommonWith<volatile T&, volatile int&>());
static_assert(!CheckCommonWith<const volatile T&, volatile int&>());
static_assert(!CheckCommonWith<const volatile T&, const int&>());
static_assert(!CheckCommonWith<const volatile T&, volatile int&>());
static_assert(!CheckCommonWith<const volatile T&, const volatile int&>());
static_assert(!CheckCommonWith<const T&, const volatile int&>());
static_assert(!CheckCommonWith<volatile T&, const volatile int&>());
static_assert(!CheckCommonWith<const volatile T&, const volatile int&>());
static_assert(!CheckCommonWith<T&, int&&>());
static_assert(!CheckCommonWith<T&, const int&&>());
static_assert(!CheckCommonWith<T&, volatile int&&>());
static_assert(!CheckCommonWith<T&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&, int&&>());
static_assert(!CheckCommonWith<volatile T&, int&&>());
static_assert(!CheckCommonWith<const volatile T&, int&&>());
static_assert(!CheckCommonWith<const T&, const int&&>());
static_assert(!CheckCommonWith<const T&, volatile int&&>());
static_assert(!CheckCommonWith<const T&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&, const int&&>());
static_assert(!CheckCommonWith<volatile T&, const int&&>());
static_assert(!CheckCommonWith<const volatile T&, const int&&>());
static_assert(!CheckCommonWith<volatile T&, const int&&>());
static_assert(!CheckCommonWith<volatile T&, volatile int&&>());
static_assert(!CheckCommonWith<volatile T&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&, volatile int&&>());
static_assert(!CheckCommonWith<volatile T&, volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&, volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&, const int&&>());
static_assert(!CheckCommonWith<const volatile T&, volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&, const volatile int&&>());
static_assert(!CheckCommonWith<volatile T&, const volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&, const volatile int&&>());
static_assert(!CheckCommonWith<T&&, int&>());
static_assert(!CheckCommonWith<T&&, const int&>());
static_assert(!CheckCommonWith<T&&, volatile int&>());
static_assert(!CheckCommonWith<T&&, const volatile int&>());
static_assert(!CheckCommonWith<const T&&, int&>());
static_assert(!CheckCommonWith<volatile T&&, int&>());
static_assert(!CheckCommonWith<const volatile T&&, int&>());
static_assert(!CheckCommonWith<const T&&, const int&>());
static_assert(!CheckCommonWith<const T&&, volatile int&>());
static_assert(!CheckCommonWith<const T&&, const volatile int&>());
static_assert(!CheckCommonWith<const T&&, const int&>());
static_assert(!CheckCommonWith<volatile T&&, const int&>());
static_assert(!CheckCommonWith<const volatile T&&, const int&>());
static_assert(!CheckCommonWith<volatile T&&, const int&>());
static_assert(!CheckCommonWith<volatile T&&, volatile int&>());
static_assert(!CheckCommonWith<volatile T&&, const volatile int&>());
static_assert(!CheckCommonWith<const T&&, volatile int&>());
static_assert(!CheckCommonWith<volatile T&&, volatile int&>());
static_assert(!CheckCommonWith<const volatile T&&, volatile int&>());
static_assert(!CheckCommonWith<const volatile T&&, const int&>());
static_assert(!CheckCommonWith<const volatile T&&, volatile int&>());
static_assert(!CheckCommonWith<const volatile T&&, const volatile int&>());
static_assert(!CheckCommonWith<const T&&, const volatile int&>());
static_assert(!CheckCommonWith<volatile T&&, const volatile int&>());
static_assert(!CheckCommonWith<const volatile T&&, const volatile int&>());
static_assert(!CheckCommonWith<T&&, int&&>());
static_assert(!CheckCommonWith<T&&, const int&&>());
static_assert(!CheckCommonWith<T&&, volatile int&&>());
static_assert(!CheckCommonWith<T&&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&&, int&&>());
static_assert(!CheckCommonWith<volatile T&&, int&&>());
static_assert(!CheckCommonWith<const volatile T&&, int&&>());
static_assert(!CheckCommonWith<const T&&, const int&&>());
static_assert(!CheckCommonWith<const T&&, volatile int&&>());
static_assert(!CheckCommonWith<const T&&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&&, const int&&>());
static_assert(!CheckCommonWith<volatile T&&, const int&&>());
static_assert(!CheckCommonWith<const volatile T&&, const int&&>());
static_assert(!CheckCommonWith<volatile T&&, const int&&>());
static_assert(!CheckCommonWith<volatile T&&, volatile int&&>());
static_assert(!CheckCommonWith<volatile T&&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&&, volatile int&&>());
static_assert(!CheckCommonWith<volatile T&&, volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&&, volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&&, const int&&>());
static_assert(!CheckCommonWith<const volatile T&&, volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&&, const volatile int&&>());
static_assert(!CheckCommonWith<const T&&, const volatile int&&>());
static_assert(!CheckCommonWith<volatile T&&, const volatile int&&>());
static_assert(!CheckCommonWith<const volatile T&&, const volatile int&&>());
} // namespace NoDefaultCommonType

struct BadBasicCommonType {
  // This test is ill-formed, NDR. If it ever blows up in our faces: that's a good thing.
  // In the meantime, the test should be included. If compiler support is added, then an include guard
  // should be placed so the test doesn't get deleted.
};

template <>
struct std::common_type<BadBasicCommonType, int> {
  using type = BadBasicCommonType;
};

template <>
struct std::common_type<int, BadBasicCommonType> {
  using type = int;
};

static_assert(requires {
  typename std::common_type_t<BadBasicCommonType, int>;
});
static_assert(requires {
  typename std::common_type_t<int, BadBasicCommonType>;
});
static_assert(!std::same_as<std::common_type_t<BadBasicCommonType, int>,
                            std::common_type_t<int, BadBasicCommonType> >);
static_assert(!CheckCommonWith<BadBasicCommonType, int>());

struct DullCommonType {};
static_assert(!std::convertible_to<DullCommonType, int>);

struct T1 {};
static_assert(!std::convertible_to<DullCommonType, T1>);

template <>
struct std::common_type<T1, int> {
  using type = DullCommonType;
};

template <>
struct std::common_type<int, T1> {
  using type = DullCommonType;
};

static_assert(HasValidCommonType<T1, int>());
static_assert(!CheckCommonWith<T1, int>());

struct CommonTypeImplicitlyConstructibleFromInt {
  explicit(false) CommonTypeImplicitlyConstructibleFromInt(int);
};
static_assert(requires {
  static_cast<CommonTypeImplicitlyConstructibleFromInt>(0);
});

struct T2 {};
static_assert(
    !std::convertible_to<CommonTypeImplicitlyConstructibleFromInt, T2>);

template <>
struct std::common_type<T2, int> {
  using type = CommonTypeImplicitlyConstructibleFromInt;
};

template <>
struct std::common_type<int, T2> {
  using type = CommonTypeImplicitlyConstructibleFromInt;
};

static_assert(HasValidCommonType<T2, int>());
static_assert(!CheckCommonWith<T2, int>());

struct CommonTypeExplicitlyConstructibleFromInt {
  explicit CommonTypeExplicitlyConstructibleFromInt(int);
};
static_assert(requires {
  static_cast<CommonTypeExplicitlyConstructibleFromInt>(0);
});

struct T3 {};
static_assert(
    !std::convertible_to<CommonTypeExplicitlyConstructibleFromInt, T2>);

template <>
struct std::common_type<T3, int> {
  using type = CommonTypeExplicitlyConstructibleFromInt;
};

template <>
struct std::common_type<int, T3> {
  using type = CommonTypeExplicitlyConstructibleFromInt;
};

static_assert(HasValidCommonType<T3, int>());
static_assert(!CheckCommonWith<T3, int>());

struct T4 {};
struct CommonTypeImplicitlyConstructibleFromT4 {
  explicit(false) CommonTypeImplicitlyConstructibleFromT4(T4);
};
static_assert(requires(T4 t4) {
  static_cast<CommonTypeImplicitlyConstructibleFromT4>(t4);
});

template <>
struct std::common_type<T4, int> {
  using type = CommonTypeImplicitlyConstructibleFromT4;
};

template <>
struct std::common_type<int, T4> {
  using type = CommonTypeImplicitlyConstructibleFromT4;
};

static_assert(HasValidCommonType<T4, int>());
static_assert(!CheckCommonWith<T4, int>());

struct T5 {};
struct CommonTypeExplicitlyConstructibleFromT5 {
  explicit CommonTypeExplicitlyConstructibleFromT5(T5);
};
static_assert(requires(T5 t5) {
  static_cast<CommonTypeExplicitlyConstructibleFromT5>(t5);
});

template <>
struct std::common_type<T5, int> {
  using type = CommonTypeExplicitlyConstructibleFromT5;
};

template <>
struct std::common_type<int, T5> {
  using type = CommonTypeExplicitlyConstructibleFromT5;
};

static_assert(HasValidCommonType<T5, int>());
static_assert(!CheckCommonWith<T5, int>());

struct T6 {};
struct CommonTypeNoCommonReference {
  CommonTypeNoCommonReference(T6);
  CommonTypeNoCommonReference(int);
};

template <>
struct std::common_type<T6, int> {
  using type = CommonTypeNoCommonReference;
};

template <>
struct std::common_type<int, T6> {
  using type = CommonTypeNoCommonReference;
};

template <>
struct std::common_type<T6&, int&> {};

template <>
struct std::common_type<int&, T6&> {};

template <>
struct std::common_type<T6&, const int&> {};

template <>
struct std::common_type<int&, const T6&> {};

template <>
struct std::common_type<T6&, volatile int&> {};

template <>
struct std::common_type<int&, volatile T6&> {};

template <>
struct std::common_type<T6&, const volatile int&> {};

template <>
struct std::common_type<int&, const volatile T6&> {};

template <>
struct std::common_type<const T6&, int&> {};

template <>
struct std::common_type<const int&, T6&> {};

template <>
struct std::common_type<const T6&, const int&> {};

template <>
struct std::common_type<const int&, const T6&> {};

template <>
struct std::common_type<const T6&, volatile int&> {};

template <>
struct std::common_type<const int&, volatile T6&> {};

template <>
struct std::common_type<const T6&, const volatile int&> {};

template <>
struct std::common_type<const int&, const volatile T6&> {};

template <>
struct std::common_type<volatile T6&, int&> {};

template <>
struct std::common_type<volatile int&, T6&> {};

template <>
struct std::common_type<volatile T6&, const int&> {};

template <>
struct std::common_type<volatile int&, const T6&> {};

template <>
struct std::common_type<volatile T6&, volatile int&> {};

template <>
struct std::common_type<volatile int&, volatile T6&> {};

template <>
struct std::common_type<volatile T6&, const volatile int&> {};

template <>
struct std::common_type<volatile int&, const volatile T6&> {};

template <>
struct std::common_type<const volatile T6&, int&> {};

template <>
struct std::common_type<const volatile int&, T6&> {};

template <>
struct std::common_type<const volatile T6&, const int&> {};

template <>
struct std::common_type<const volatile int&, const T6&> {};

template <>
struct std::common_type<const volatile T6&, volatile int&> {};

template <>
struct std::common_type<const volatile int&, volatile T6&> {};

template <>
struct std::common_type<const volatile T6&, const volatile int&> {};

template <>
struct std::common_type<const volatile int&, const volatile T6&> {};

template <typename T, typename U>
constexpr bool HasCommonReference() noexcept {
  return requires { typename std::common_reference_t<T, U>; };
}

static_assert(HasValidCommonType<T6, int>());
static_assert(!HasCommonReference<const T6&, const int&>());
static_assert(!CheckCommonWith<T6, int>());

struct T7 {};
struct CommonTypeNoMetaCommonReference {
  CommonTypeNoMetaCommonReference(T7);
  CommonTypeNoMetaCommonReference(int);
};

template <>
struct std::common_type<T7, int> {
  using type = CommonTypeNoMetaCommonReference;
};

template <>
struct std::common_type<int, T7> {
  using type = CommonTypeNoMetaCommonReference;
};

template <>
struct std::common_type<T7&, int&> {
  using type = void;
};

template <>
struct std::common_type<int&, T7&> {
  using type = void;
};

template <>
struct std::common_type<T7&, const int&> {
  using type = void;
};

template <>
struct std::common_type<int&, const T7&> {
  using type = void;
};

template <>
struct std::common_type<T7&, volatile int&> {
  using type = void;
};

template <>
struct std::common_type<int&, volatile T7&> {
  using type = void;
};

template <>
struct std::common_type<T7&, const volatile int&> {
  using type = void;
};

template <>
struct std::common_type<int&, const volatile T7&> {
  using type = void;
};

template <>
struct std::common_type<const T7&, int&> {
  using type = void;
};

template <>
struct std::common_type<const int&, T7&> {
  using type = void;
};

template <>
struct std::common_type<const T7&, const int&> {
  using type = void;
};

template <>
struct std::common_type<const int&, const T7&> {
  using type = void;
};

template <>
struct std::common_type<const T7&, volatile int&> {
  using type = void;
};

template <>
struct std::common_type<const int&, volatile T7&> {
  using type = void;
};

template <>
struct std::common_type<const T7&, const volatile int&> {
  using type = void;
};

template <>
struct std::common_type<const int&, const volatile T7&> {
  using type = void;
};

template <>
struct std::common_type<volatile T7&, int&> {
  using type = void;
};

template <>
struct std::common_type<volatile int&, T7&> {
  using type = void;
};

template <>
struct std::common_type<volatile T7&, const int&> {
  using type = void;
};

template <>
struct std::common_type<volatile int&, const T7&> {
  using type = void;
};

template <>
struct std::common_type<volatile T7&, volatile int&> {
  using type = void;
};

template <>
struct std::common_type<volatile int&, volatile T7&> {
  using type = void;
};

template <>
struct std::common_type<volatile T7&, const volatile int&> {
  using type = void;
};

template <>
struct std::common_type<volatile int&, const volatile T7&> {
  using type = void;
};

template <>
struct std::common_type<const volatile T7&, int&> {
  using type = void;
};

template <>
struct std::common_type<const volatile int&, T7&> {
  using type = void;
};

template <>
struct std::common_type<const volatile T7&, const int&> {
  using type = void;
};

template <>
struct std::common_type<const volatile int&, const T7&> {
  using type = void;
};

template <>
struct std::common_type<const volatile T7&, volatile int&> {
  using type = void;
};

template <>
struct std::common_type<const volatile int&, volatile T7&> {
  using type = void;
};

template <>
struct std::common_type<const volatile T7&, const volatile int&> {
  using type = void;
};

template <>
struct std::common_type<const volatile int&, const volatile T7&> {
  using type = void;
};

static_assert(HasValidCommonType<T7, int>());
static_assert(HasValidCommonType<const T7&, const int&>());
static_assert(HasCommonReference<const T7&, const int&>());
static_assert(
    !HasCommonReference<std::common_type_t<T7, int>&,
                        std::common_reference_t<const T7&, const int&> >());
static_assert(!CheckCommonWith<T7, int>());

struct CommonWithInt {
  operator int() const volatile;
};

template <>
struct std::common_type<CommonWithInt, int> {
  using type = int;
};

template <>
struct std::common_type<int, CommonWithInt> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<CommonWithInt&, int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<int&, CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<CommonWithInt&, const int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<int&, const CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<CommonWithInt&, volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<int&, volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<CommonWithInt&, const volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<int&, const volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const CommonWithInt&, int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const int&, CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const CommonWithInt&, const int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const int&, const CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const CommonWithInt&, volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const int&, volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const CommonWithInt&, const volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const int&, const volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile CommonWithInt&, int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile int&, CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile CommonWithInt&, const int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile int&, const CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile CommonWithInt&, volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile int&, volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile CommonWithInt&, const volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<volatile int&, const volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile CommonWithInt&, int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile int&, CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile CommonWithInt&, const int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile int&, const CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile CommonWithInt&, volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile int&, volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile CommonWithInt&, const volatile int&> : std::common_type<CommonWithInt, int> {};

template <>
struct std::common_type<const volatile int&, const volatile CommonWithInt&> : std::common_type<CommonWithInt, int> {};

static_assert(CheckCommonWith<CommonWithInt, int>());

struct CommonWithIntButRefLong {
  operator int() const volatile;
};

template <>
struct std::common_type<CommonWithIntButRefLong, int> {
  using type = int;
};

template <>
struct std::common_type<int, CommonWithIntButRefLong> : std::common_type<CommonWithIntButRefLong, int> {};

template <>
struct std::common_type<CommonWithIntButRefLong&, int&> {
  using type = long;
};

template <>
struct std::common_type<int&, CommonWithIntButRefLong&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<CommonWithIntButRefLong&, const int&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<int&, const CommonWithIntButRefLong&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<CommonWithIntButRefLong&, volatile int&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<int&, volatile CommonWithIntButRefLong&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<CommonWithIntButRefLong&, const volatile int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<int&, const volatile CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const CommonWithIntButRefLong&, int&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const int&, CommonWithIntButRefLong&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const CommonWithIntButRefLong&, const int&> : std::common_type<CommonWithIntButRefLong&, int&> {
};

template <>
struct std::common_type<const int&, const CommonWithIntButRefLong&> : std::common_type<CommonWithIntButRefLong&, int&> {
};

template <>
struct std::common_type<const CommonWithIntButRefLong&, volatile int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const int&, volatile CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const CommonWithIntButRefLong&, const volatile int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const int&, const volatile CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile CommonWithIntButRefLong&, int&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile int&, CommonWithIntButRefLong&> : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile CommonWithIntButRefLong&, const int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile int&, const CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile CommonWithIntButRefLong&, volatile int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile int&, volatile CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile CommonWithIntButRefLong&, const volatile int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<volatile int&, const volatile CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile CommonWithIntButRefLong&, int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile int&, CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile CommonWithIntButRefLong&, const int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile int&, const CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile CommonWithIntButRefLong&, volatile int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile int&, volatile CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile CommonWithIntButRefLong&, const volatile int&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct std::common_type<const volatile int&, const volatile CommonWithIntButRefLong&>
    : std::common_type<CommonWithIntButRefLong&, int&> {};

static_assert(CheckCommonWith<CommonWithIntButRefLong, int>());
