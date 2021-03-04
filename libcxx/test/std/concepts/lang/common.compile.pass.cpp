//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class From, class To>
// concept common_with;

#include <concepts>

template <class T, class U>
constexpr bool CheckCommonWith() noexcept {
  constexpr auto result = std::common_with<T, U>;
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
#ifndef _LIBCPP_HAS_NO_INT128
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

namespace std {
template <>
struct common_type<BadBasicCommonType, int> {
  using type = BadBasicCommonType;
};

template <>
struct common_type<int, BadBasicCommonType> {
  using type = int;
};
} // namespace std
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

namespace std {
template <>
struct common_type<T1, int> {
  using type = DullCommonType;
};

template <>
struct common_type<int, T1> {
  using type = DullCommonType;
};
} // namespace std
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

namespace std {
template <>
struct common_type<T2, int> {
  using type = CommonTypeImplicitlyConstructibleFromInt;
};

template <>
struct common_type<int, T2> {
  using type = CommonTypeImplicitlyConstructibleFromInt;
};
} // namespace std
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

namespace std {
template <>
struct common_type<T3, int> {
  using type = CommonTypeExplicitlyConstructibleFromInt;
};

template <>
struct common_type<int, T3> {
  using type = CommonTypeExplicitlyConstructibleFromInt;
};
} // namespace std
static_assert(HasValidCommonType<T3, int>());
static_assert(!CheckCommonWith<T3, int>());

struct T4 {};
struct CommonTypeImplicitlyConstructibleFromT4 {
  explicit(false) CommonTypeImplicitlyConstructibleFromT4(T4);
};
static_assert(requires(T4 t4) {
  static_cast<CommonTypeImplicitlyConstructibleFromT4>(t4);
});

namespace std {
template <>
struct common_type<T4, int> {
  using type = CommonTypeImplicitlyConstructibleFromT4;
};

template <>
struct common_type<int, T4> {
  using type = CommonTypeImplicitlyConstructibleFromT4;
};
} // namespace std
static_assert(HasValidCommonType<T4, int>());
static_assert(!CheckCommonWith<T4, int>());

struct T5 {};
struct CommonTypeExplicitlyConstructibleFromT5 {
  explicit CommonTypeExplicitlyConstructibleFromT5(T5);
};
static_assert(requires(T5 t5) {
  static_cast<CommonTypeExplicitlyConstructibleFromT5>(t5);
});

namespace std {
template <>
struct common_type<T5, int> {
  using type = CommonTypeExplicitlyConstructibleFromT5;
};

template <>
struct common_type<int, T5> {
  using type = CommonTypeExplicitlyConstructibleFromT5;
};
} // namespace std
static_assert(HasValidCommonType<T5, int>());
static_assert(!CheckCommonWith<T5, int>());

struct T6 {};
struct CommonTypeNoCommonReference {
  CommonTypeNoCommonReference(T6);
  CommonTypeNoCommonReference(int);
};

namespace std {
template <>
struct common_type<T6, int> {
  using type = CommonTypeNoCommonReference;
};

template <>
struct common_type<int, T6> {
  using type = CommonTypeNoCommonReference;
};

template <>
struct common_type<T6&, int&> {};

template <>
struct common_type<int&, T6&> {};

template <>
struct common_type<T6&, const int&> {};

template <>
struct common_type<int&, const T6&> {};

template <>
struct common_type<T6&, volatile int&> {};

template <>
struct common_type<int&, volatile T6&> {};

template <>
struct common_type<T6&, const volatile int&> {};

template <>
struct common_type<int&, const volatile T6&> {};

template <>
struct common_type<const T6&, int&> {};

template <>
struct common_type<const int&, T6&> {};

template <>
struct common_type<const T6&, const int&> {};

template <>
struct common_type<const int&, const T6&> {};

template <>
struct common_type<const T6&, volatile int&> {};

template <>
struct common_type<const int&, volatile T6&> {};

template <>
struct common_type<const T6&, const volatile int&> {};

template <>
struct common_type<const int&, const volatile T6&> {};

template <>
struct common_type<volatile T6&, int&> {};

template <>
struct common_type<volatile int&, T6&> {};

template <>
struct common_type<volatile T6&, const int&> {};

template <>
struct common_type<volatile int&, const T6&> {};

template <>
struct common_type<volatile T6&, volatile int&> {};

template <>
struct common_type<volatile int&, volatile T6&> {};

template <>
struct common_type<volatile T6&, const volatile int&> {};

template <>
struct common_type<volatile int&, const volatile T6&> {};

template <>
struct common_type<const volatile T6&, int&> {};

template <>
struct common_type<const volatile int&, T6&> {};

template <>
struct common_type<const volatile T6&, const int&> {};

template <>
struct common_type<const volatile int&, const T6&> {};

template <>
struct common_type<const volatile T6&, volatile int&> {};

template <>
struct common_type<const volatile int&, volatile T6&> {};

template <>
struct common_type<const volatile T6&, const volatile int&> {};

template <>
struct common_type<const volatile int&, const volatile T6&> {};
} // namespace std

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

namespace std {
template <>
struct common_type<T7, int> {
  using type = CommonTypeNoMetaCommonReference;
};

template <>
struct common_type<int, T7> {
  using type = CommonTypeNoMetaCommonReference;
};

template <>
struct common_type<T7&, int&> {
  using type = void;
};

template <>
struct common_type<int&, T7&> {
  using type = void;
};

template <>
struct common_type<T7&, const int&> {
  using type = void;
};

template <>
struct common_type<int&, const T7&> {
  using type = void;
};

template <>
struct common_type<T7&, volatile int&> {
  using type = void;
};

template <>
struct common_type<int&, volatile T7&> {
  using type = void;
};

template <>
struct common_type<T7&, const volatile int&> {
  using type = void;
};

template <>
struct common_type<int&, const volatile T7&> {
  using type = void;
};

template <>
struct common_type<const T7&, int&> {
  using type = void;
};

template <>
struct common_type<const int&, T7&> {
  using type = void;
};

template <>
struct common_type<const T7&, const int&> {
  using type = void;
};

template <>
struct common_type<const int&, const T7&> {
  using type = void;
};

template <>
struct common_type<const T7&, volatile int&> {
  using type = void;
};

template <>
struct common_type<const int&, volatile T7&> {
  using type = void;
};

template <>
struct common_type<const T7&, const volatile int&> {
  using type = void;
};

template <>
struct common_type<const int&, const volatile T7&> {
  using type = void;
};

template <>
struct common_type<volatile T7&, int&> {
  using type = void;
};

template <>
struct common_type<volatile int&, T7&> {
  using type = void;
};

template <>
struct common_type<volatile T7&, const int&> {
  using type = void;
};

template <>
struct common_type<volatile int&, const T7&> {
  using type = void;
};

template <>
struct common_type<volatile T7&, volatile int&> {
  using type = void;
};

template <>
struct common_type<volatile int&, volatile T7&> {
  using type = void;
};

template <>
struct common_type<volatile T7&, const volatile int&> {
  using type = void;
};

template <>
struct common_type<volatile int&, const volatile T7&> {
  using type = void;
};

template <>
struct common_type<const volatile T7&, int&> {
  using type = void;
};

template <>
struct common_type<const volatile int&, T7&> {
  using type = void;
};

template <>
struct common_type<const volatile T7&, const int&> {
  using type = void;
};

template <>
struct common_type<const volatile int&, const T7&> {
  using type = void;
};

template <>
struct common_type<const volatile T7&, volatile int&> {
  using type = void;
};

template <>
struct common_type<const volatile int&, volatile T7&> {
  using type = void;
};

template <>
struct common_type<const volatile T7&, const volatile int&> {
  using type = void;
};

template <>
struct common_type<const volatile int&, const volatile T7&> {
  using type = void;
};
} // namespace std
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

namespace std {
template <>
struct common_type<CommonWithInt, int> {
  using type = int;
};

template <>
struct common_type<int, CommonWithInt> : common_type<CommonWithInt, int> {};

template <>
struct common_type<CommonWithInt&, int&> : common_type<CommonWithInt, int> {};

template <>
struct common_type<int&, CommonWithInt&> : common_type<CommonWithInt, int> {};

template <>
struct common_type<CommonWithInt&, const int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<int&, const CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<CommonWithInt&, volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<int&, volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<CommonWithInt&, const volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<int&, const volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const CommonWithInt&, int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const int&, CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const CommonWithInt&, const int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const int&, const CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const CommonWithInt&, volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const int&, volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const CommonWithInt&, const volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const int&, const volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile CommonWithInt&, int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile int&, CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile CommonWithInt&, const int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile int&, const CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile CommonWithInt&, volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile int&, volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile CommonWithInt&, const volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<volatile int&, const volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile CommonWithInt&, int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile int&, CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile CommonWithInt&, const int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile int&, const CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile CommonWithInt&, volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile int&, volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile CommonWithInt&, const volatile int&>
    : common_type<CommonWithInt, int> {};

template <>
struct common_type<const volatile int&, const volatile CommonWithInt&>
    : common_type<CommonWithInt, int> {};
} // namespace std
static_assert(CheckCommonWith<CommonWithInt, int>());

struct CommonWithIntButRefLong {
  operator int() const volatile;
};

namespace std {
template <>
struct common_type<CommonWithIntButRefLong, int> {
  using type = int;
};

template <>
struct common_type<int, CommonWithIntButRefLong>
    : common_type<CommonWithIntButRefLong, int> {};

template <>
struct common_type<CommonWithIntButRefLong&, int&> {
  using type = long;
};

template <>
struct common_type<int&, CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<CommonWithIntButRefLong&, const int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<int&, const CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<CommonWithIntButRefLong&, volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<int&, volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<CommonWithIntButRefLong&, const volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<int&, const volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const CommonWithIntButRefLong&, int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const int&, CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const CommonWithIntButRefLong&, const int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const int&, const CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const CommonWithIntButRefLong&, volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const int&, volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const CommonWithIntButRefLong&, const volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const int&, const volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile CommonWithIntButRefLong&, int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile int&, CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile CommonWithIntButRefLong&, const int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile int&, const CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile CommonWithIntButRefLong&, volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile int&, volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile CommonWithIntButRefLong&, const volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<volatile int&, const volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile CommonWithIntButRefLong&, int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile int&, CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile CommonWithIntButRefLong&, const int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile int&, const CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile CommonWithIntButRefLong&, volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile int&, volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile CommonWithIntButRefLong&, const volatile int&>
    : common_type<CommonWithIntButRefLong&, int&> {};

template <>
struct common_type<const volatile int&, const volatile CommonWithIntButRefLong&>
    : common_type<CommonWithIntButRefLong&, int&> {};
} // namespace std
static_assert(CheckCommonWith<CommonWithIntButRefLong, int>());

int main(int, char**) { return 0; }
