//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class From, class To>
// concept common_reference_with;

#include <concepts>
#include <type_traits>

#include "test_macros.h"

template <class T, class U>
constexpr bool CheckCommonReferenceWith() noexcept {
  static_assert(std::common_reference_with<T, U&>);
  static_assert(std::common_reference_with<T, const U&>);
  static_assert(std::common_reference_with<T, volatile U&>);
  static_assert(std::common_reference_with<T, const volatile U&>);
  static_assert(std::common_reference_with<T, U&&>);
  static_assert(std::common_reference_with<T, const U&&>);
  static_assert(std::common_reference_with<T, volatile U&&>);
  static_assert(std::common_reference_with<T, const volatile U&&>);
  static_assert(std::common_reference_with<T&, U&&>);
  static_assert(std::common_reference_with<T&, const U&&>);
  static_assert(std::common_reference_with<T&, volatile U&&>);
  static_assert(std::common_reference_with<T&, const volatile U&&>);
  static_assert(std::common_reference_with<const T&, U&&>);
  static_assert(std::common_reference_with<const T&, const U&&>);
  static_assert(std::common_reference_with<const T&, volatile U&&>);
  static_assert(std::common_reference_with<const T&, const volatile U&&>);
  static_assert(std::common_reference_with<volatile T&, U&&>);
  static_assert(std::common_reference_with<volatile T&, const U&&>);
  static_assert(std::common_reference_with<volatile T&, volatile U&&>);
  static_assert(std::common_reference_with<volatile T&, const volatile U&&>);
  static_assert(std::common_reference_with<const volatile T&, U&&>);
  static_assert(std::common_reference_with<const volatile T&, const U&&>);
  static_assert(std::common_reference_with<const volatile T&, volatile U&&>);
  static_assert(
      std::common_reference_with<const volatile T&, const volatile U&&>);

  return std::common_reference_with<T, U>;
}

namespace BuiltinTypes {
// fundamental types
static_assert(std::common_reference_with<void, void>);
static_assert(CheckCommonReferenceWith<int, int>());
static_assert(CheckCommonReferenceWith<int, long>());
static_assert(CheckCommonReferenceWith<int, unsigned char>());
#ifndef TEST_HAS_NO_INT128
static_assert(CheckCommonReferenceWith<int, __int128_t>());
#endif
static_assert(CheckCommonReferenceWith<int, double>());

// arrays
static_assert(CheckCommonReferenceWith<int[5], int[5]>());

// pointers (common with void*)
static_assert(CheckCommonReferenceWith<int*, void*>());
static_assert(CheckCommonReferenceWith<int*, const void*>());
static_assert(CheckCommonReferenceWith<int*, volatile void*>());
static_assert(CheckCommonReferenceWith<int*, const volatile void*>());
static_assert(CheckCommonReferenceWith<const int*, void*>());
static_assert(CheckCommonReferenceWith<const int*, const void*>());
static_assert(CheckCommonReferenceWith<const int*, volatile void*>());
static_assert(CheckCommonReferenceWith<const int*, const volatile void*>());
static_assert(CheckCommonReferenceWith<volatile int*, void*>());
static_assert(CheckCommonReferenceWith<volatile int*, const void*>());
static_assert(CheckCommonReferenceWith<volatile int*, volatile void*>());
static_assert(CheckCommonReferenceWith<volatile int*, const volatile void*>());
static_assert(CheckCommonReferenceWith<const volatile int*, void*>());
static_assert(CheckCommonReferenceWith<const volatile int*, const void*>());
static_assert(CheckCommonReferenceWith<const volatile int*, volatile void*>());
static_assert(
    CheckCommonReferenceWith<const volatile int*, const volatile void*>());

static_assert(CheckCommonReferenceWith<int (*)(), int (*)()>());
static_assert(CheckCommonReferenceWith<int (*)(), int (*)() noexcept>());
struct S {};
static_assert(CheckCommonReferenceWith<int S::*, int S::*>());
static_assert(CheckCommonReferenceWith<int S::*, const int S::*>());
static_assert(CheckCommonReferenceWith<int (S::*)(), int (S::*)()>());
static_assert(CheckCommonReferenceWith<int (S::*)(), int (S::*)() noexcept>());
static_assert(
    CheckCommonReferenceWith<int (S::*)() const, int (S::*)() const>());
static_assert(CheckCommonReferenceWith<int (S::*)() const,
                                       int (S::*)() const noexcept>());
static_assert(
    CheckCommonReferenceWith<int (S::*)() volatile, int (S::*)() volatile>());
static_assert(CheckCommonReferenceWith<int (S::*)() volatile,
                                       int (S::*)() volatile noexcept>());
static_assert(CheckCommonReferenceWith<int (S::*)() const volatile,
                                       int (S::*)() const volatile>());
static_assert(CheckCommonReferenceWith<int (S::*)() const volatile,
                                       int (S::*)() const volatile noexcept>());

// nonsense
static_assert(!std::common_reference_with<double, float*>);
static_assert(!std::common_reference_with<int, int[5]>);
static_assert(!std::common_reference_with<int*, long*>);
static_assert(!std::common_reference_with<int*, unsigned int*>);
static_assert(!std::common_reference_with<int (*)(), int (*)(int)>);
static_assert(!std::common_reference_with<int S::*, float S::*>);
static_assert(!std::common_reference_with<int (S::*)(), int (S::*)() const>);
static_assert(!std::common_reference_with<int (S::*)(), int (S::*)() volatile>);
static_assert(
    !std::common_reference_with<int (S::*)(), int (S::*)() const volatile>);
static_assert(
    !std::common_reference_with<int (S::*)() const, int (S::*)() volatile>);
static_assert(!std::common_reference_with<int (S::*)() const,
                                          int (S::*)() const volatile>);
static_assert(!std::common_reference_with<int (S::*)() volatile,
                                          int (S::*)() const volatile>);
} // namespace BuiltinTypes

namespace NoDefaultCommonReference {
class T {};

static_assert(!std::common_reference_with<T, int>);
static_assert(!std::common_reference_with<int, T>);
static_assert(!std::common_reference_with<T, int[10]>);
static_assert(!std::common_reference_with<T[10], int>);
static_assert(!std::common_reference_with<T*, int*>);
static_assert(!std::common_reference_with<T*, const int*>);
static_assert(!std::common_reference_with<T*, volatile int*>);
static_assert(!std::common_reference_with<T*, const volatile int*>);
static_assert(!std::common_reference_with<const T*, int*>);
static_assert(!std::common_reference_with<volatile T*, int*>);
static_assert(!std::common_reference_with<const volatile T*, int*>);
static_assert(!std::common_reference_with<const T*, const int*>);
static_assert(!std::common_reference_with<const T*, volatile int*>);
static_assert(!std::common_reference_with<const T*, const volatile int*>);
static_assert(!std::common_reference_with<const T*, const int*>);
static_assert(!std::common_reference_with<volatile T*, const int*>);
static_assert(!std::common_reference_with<const volatile T*, const int*>);
static_assert(!std::common_reference_with<volatile T*, const int*>);
static_assert(!std::common_reference_with<volatile T*, volatile int*>);
static_assert(!std::common_reference_with<volatile T*, const volatile int*>);
static_assert(!std::common_reference_with<const T*, volatile int*>);
static_assert(!std::common_reference_with<volatile T*, volatile int*>);
static_assert(!std::common_reference_with<const volatile T*, volatile int*>);
static_assert(!std::common_reference_with<const volatile T*, const int*>);
static_assert(!std::common_reference_with<const volatile T*, volatile int*>);
static_assert(
    !std::common_reference_with<const volatile T*, const volatile int*>);
static_assert(!std::common_reference_with<const T*, const volatile int*>);
static_assert(!std::common_reference_with<volatile T*, const volatile int*>);
static_assert(
    !std::common_reference_with<const volatile T*, const volatile int*>);
static_assert(!std::common_reference_with<T&, int&>);
static_assert(!std::common_reference_with<T&, const int&>);
static_assert(!std::common_reference_with<T&, volatile int&>);
static_assert(!std::common_reference_with<T&, const volatile int&>);
static_assert(!std::common_reference_with<const T&, int&>);
static_assert(!std::common_reference_with<volatile T&, int&>);
static_assert(!std::common_reference_with<const volatile T&, int&>);
static_assert(!std::common_reference_with<const T&, const int&>);
static_assert(!std::common_reference_with<const T&, volatile int&>);
static_assert(!std::common_reference_with<const T&, const volatile int&>);
static_assert(!std::common_reference_with<const T&, const int&>);
static_assert(!std::common_reference_with<volatile T&, const int&>);
static_assert(!std::common_reference_with<const volatile T&, const int&>);
static_assert(!std::common_reference_with<volatile T&, const int&>);
static_assert(!std::common_reference_with<volatile T&, volatile int&>);
static_assert(!std::common_reference_with<volatile T&, const volatile int&>);
static_assert(!std::common_reference_with<const T&, volatile int&>);
static_assert(!std::common_reference_with<volatile T&, volatile int&>);
static_assert(!std::common_reference_with<const volatile T&, volatile int&>);
static_assert(!std::common_reference_with<const volatile T&, const int&>);
static_assert(!std::common_reference_with<const volatile T&, volatile int&>);
static_assert(
    !std::common_reference_with<const volatile T&, const volatile int&>);
static_assert(!std::common_reference_with<const T&, const volatile int&>);
static_assert(!std::common_reference_with<volatile T&, const volatile int&>);
static_assert(
    !std::common_reference_with<const volatile T&, const volatile int&>);
static_assert(!std::common_reference_with<T&, int&&>);
static_assert(!std::common_reference_with<T&, const int&&>);
static_assert(!std::common_reference_with<T&, volatile int&&>);
static_assert(!std::common_reference_with<T&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&, int&&>);
static_assert(!std::common_reference_with<volatile T&, int&&>);
static_assert(!std::common_reference_with<const volatile T&, int&&>);
static_assert(!std::common_reference_with<const T&, const int&&>);
static_assert(!std::common_reference_with<const T&, volatile int&&>);
static_assert(!std::common_reference_with<const T&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&, const int&&>);
static_assert(!std::common_reference_with<volatile T&, const int&&>);
static_assert(!std::common_reference_with<const volatile T&, const int&&>);
static_assert(!std::common_reference_with<volatile T&, const int&&>);
static_assert(!std::common_reference_with<volatile T&, volatile int&&>);
static_assert(!std::common_reference_with<volatile T&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&, volatile int&&>);
static_assert(!std::common_reference_with<volatile T&, volatile int&&>);
static_assert(!std::common_reference_with<const volatile T&, volatile int&&>);
static_assert(!std::common_reference_with<const volatile T&, const int&&>);
static_assert(!std::common_reference_with<const volatile T&, volatile int&&>);
static_assert(
    !std::common_reference_with<const volatile T&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&, const volatile int&&>);
static_assert(!std::common_reference_with<volatile T&, const volatile int&&>);
static_assert(
    !std::common_reference_with<const volatile T&, const volatile int&&>);
static_assert(!std::common_reference_with<T&&, int&>);
static_assert(!std::common_reference_with<T&&, const int&>);
static_assert(!std::common_reference_with<T&&, volatile int&>);
static_assert(!std::common_reference_with<T&&, const volatile int&>);
static_assert(!std::common_reference_with<const T&&, int&>);
static_assert(!std::common_reference_with<volatile T&&, int&>);
static_assert(!std::common_reference_with<const volatile T&&, int&>);
static_assert(!std::common_reference_with<const T&&, const int&>);
static_assert(!std::common_reference_with<const T&&, volatile int&>);
static_assert(!std::common_reference_with<const T&&, const volatile int&>);
static_assert(!std::common_reference_with<const T&&, const int&>);
static_assert(!std::common_reference_with<volatile T&&, const int&>);
static_assert(!std::common_reference_with<const volatile T&&, const int&>);
static_assert(!std::common_reference_with<volatile T&&, const int&>);
static_assert(!std::common_reference_with<volatile T&&, volatile int&>);
static_assert(!std::common_reference_with<volatile T&&, const volatile int&>);
static_assert(!std::common_reference_with<const T&&, volatile int&>);
static_assert(!std::common_reference_with<volatile T&&, volatile int&>);
static_assert(!std::common_reference_with<const volatile T&&, volatile int&>);
static_assert(!std::common_reference_with<const volatile T&&, const int&>);
static_assert(!std::common_reference_with<const volatile T&&, volatile int&>);
static_assert(
    !std::common_reference_with<const volatile T&&, const volatile int&>);
static_assert(!std::common_reference_with<const T&&, const volatile int&>);
static_assert(!std::common_reference_with<volatile T&&, const volatile int&>);
static_assert(
    !std::common_reference_with<const volatile T&&, const volatile int&>);
static_assert(!std::common_reference_with<T&&, int&&>);
static_assert(!std::common_reference_with<T&&, const int&&>);
static_assert(!std::common_reference_with<T&&, volatile int&&>);
static_assert(!std::common_reference_with<T&&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&&, int&&>);
static_assert(!std::common_reference_with<volatile T&&, int&&>);
static_assert(!std::common_reference_with<const volatile T&&, int&&>);
static_assert(!std::common_reference_with<const T&&, const int&&>);
static_assert(!std::common_reference_with<const T&&, volatile int&&>);
static_assert(!std::common_reference_with<const T&&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&&, const int&&>);
static_assert(!std::common_reference_with<volatile T&&, const int&&>);
static_assert(!std::common_reference_with<const volatile T&&, const int&&>);
static_assert(!std::common_reference_with<volatile T&&, const int&&>);
static_assert(!std::common_reference_with<volatile T&&, volatile int&&>);
static_assert(!std::common_reference_with<volatile T&&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&&, volatile int&&>);
static_assert(!std::common_reference_with<volatile T&&, volatile int&&>);
static_assert(!std::common_reference_with<const volatile T&&, volatile int&&>);
static_assert(!std::common_reference_with<const volatile T&&, const int&&>);
static_assert(!std::common_reference_with<const volatile T&&, volatile int&&>);
static_assert(
    !std::common_reference_with<const volatile T&&, const volatile int&&>);
static_assert(!std::common_reference_with<const T&&, const volatile int&&>);
static_assert(!std::common_reference_with<volatile T&&, const volatile int&&>);
static_assert(
    !std::common_reference_with<const volatile T&&, const volatile int&&>);
} // namespace NoDefaultCommonReference

struct BadBasicCommonReference {
  // This test is ill-formed, NDR. If it ever blows up in our faces: that's a good thing.
  // In the meantime, the test should be included. If compiler support is added, then an include guard
  // should be placed so the test doesn't get deleted.
  operator int() const;
  operator int&();
};
static_assert(std::convertible_to<BadBasicCommonReference, int>);
static_assert(std::convertible_to<BadBasicCommonReference, int&>);

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<BadBasicCommonReference, int, X, Y> {
  using type = BadBasicCommonReference&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<int, BadBasicCommonReference, X, Y> {
  using type = int&;
};

static_assert(!std::common_reference_with<BadBasicCommonReference, int>);

struct StructNotConvertibleToCommonReference {
  explicit(false) StructNotConvertibleToCommonReference(int);
};
static_assert(std::convertible_to<int, StructNotConvertibleToCommonReference>);

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<StructNotConvertibleToCommonReference, int, X, Y> {
  using type = int&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<int, StructNotConvertibleToCommonReference, X, Y> {
  using type = int&;
};

static_assert(
    !std::common_reference_with<StructNotConvertibleToCommonReference, int>);

struct IntNotConvertibleToCommonReference {
  operator int&() const;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<IntNotConvertibleToCommonReference, int, X, Y> {
  using type = int&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<int, IntNotConvertibleToCommonReference, X, Y> {
  using type = int&;
};

static_assert(
    !std::common_reference_with<StructNotConvertibleToCommonReference, int>);

struct HasCommonReference {
  explicit(false) HasCommonReference(int);
  operator int&() const;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<HasCommonReference, int, X, Y> {
  using type = int&;
};

template <template <class> class X, template <class> class Y>
struct std::basic_common_reference<int, HasCommonReference, X, Y> {
  using type = int&;
};

static_assert(!std::common_reference_with<HasCommonReference, int>);
static_assert(std::common_reference_with<HasCommonReference, int&>);
