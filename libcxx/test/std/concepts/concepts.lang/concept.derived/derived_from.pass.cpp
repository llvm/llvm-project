//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class Derived, class Base>
// concept derived_from;

#include <concepts>
#include <type_traits>

struct Base1 {};
struct Derived1 : Base1 {};
struct Derived2 : Base1 {};

struct DerivedPrivate : private Base1 {};
struct Derived3 : DerivedPrivate {};

struct DerivedProtected : protected DerivedPrivate {};
struct Derived4 : DerivedProtected {};
struct Derived5 : Derived4 {};

template <typename From, typename To>
constexpr void CheckNotDerivedFromPointer() {
  { // From as pointer
    static_assert(!std::derived_from<From*, To>);
    static_assert(!std::derived_from<From*, const To>);
    static_assert(!std::derived_from<From*, volatile To>);
    static_assert(!std::derived_from<From*, const volatile To>);

    if constexpr (!std::same_as<To, void>) {
      static_assert(!std::derived_from<From*, To&>);
      static_assert(!std::derived_from<From*, const To&>);
      static_assert(!std::derived_from<From*, volatile To&>);
      static_assert(!std::derived_from<From*, const volatile To&>);

      static_assert(!std::derived_from<From*, To&&>);
      static_assert(!std::derived_from<From*, const To&&>);
      static_assert(!std::derived_from<From*, volatile To&&>);
      static_assert(!std::derived_from<From*, const volatile To&&>);

      static_assert(!std::derived_from<const From*, To&>);
      static_assert(!std::derived_from<const From*, const To&>);
      static_assert(!std::derived_from<const From*, volatile To&>);
      static_assert(!std::derived_from<const From*, const volatile To&>);

      static_assert(!std::derived_from<const From*, To&&>);
      static_assert(!std::derived_from<const From*, const To&&>);
      static_assert(!std::derived_from<const From*, volatile To&&>);
      static_assert(!std::derived_from<const From*, const volatile To&&>);

      static_assert(!std::derived_from<volatile From*, To&>);
      static_assert(!std::derived_from<volatile From*, const To&>);
      static_assert(!std::derived_from<volatile From*, volatile To&>);
      static_assert(!std::derived_from<volatile From*, const volatile To&>);

      static_assert(!std::derived_from<volatile From*, To&&>);
      static_assert(!std::derived_from<volatile From*, const To&&>);
      static_assert(!std::derived_from<volatile From*, volatile To&&>);
      static_assert(!std::derived_from<volatile From*, const volatile To&&>);

      static_assert(!std::derived_from<const volatile From*, To&>);
      static_assert(!std::derived_from<const volatile From*, const To&>);
      static_assert(!std::derived_from<const volatile From*, volatile To&>);
      static_assert(
          !std::derived_from<const volatile From*, const volatile To&>);

      static_assert(!std::derived_from<const volatile From*, To&&>);
      static_assert(!std::derived_from<const volatile From*, const To&&>);
      static_assert(!std::derived_from<const volatile From*, volatile To&&>);
      static_assert(
          !std::derived_from<const volatile From*, const volatile To&&>);
    }
  }
  { // To as pointer
    static_assert(!std::derived_from<From, To*>);
    static_assert(!std::derived_from<From, const To*>);
    static_assert(!std::derived_from<From, volatile To*>);
    static_assert(!std::derived_from<From, const volatile To*>);

    if constexpr (!std::same_as<From, void>) {
      static_assert(!std::derived_from<From&, To*>);
      static_assert(!std::derived_from<From&, const To*>);
      static_assert(!std::derived_from<From&, volatile To*>);
      static_assert(!std::derived_from<From&, const volatile To*>);

      static_assert(!std::derived_from<From&&, To*>);
      static_assert(!std::derived_from<From&&, const To*>);
      static_assert(!std::derived_from<From&&, volatile To*>);
      static_assert(!std::derived_from<From&&, const volatile To*>);

      static_assert(!std::derived_from<const From&, To*>);
      static_assert(!std::derived_from<const From&, const To*>);
      static_assert(!std::derived_from<const From&, volatile To*>);
      static_assert(!std::derived_from<const From&, const volatile To*>);

      static_assert(!std::derived_from<const From&&, To*>);
      static_assert(!std::derived_from<const From&&, const To*>);
      static_assert(!std::derived_from<const From&&, volatile To*>);
      static_assert(!std::derived_from<const From&&, const volatile To*>);

      static_assert(!std::derived_from<volatile From&, To*>);
      static_assert(!std::derived_from<volatile From&, const To*>);
      static_assert(!std::derived_from<volatile From&, volatile To*>);
      static_assert(!std::derived_from<volatile From&, const volatile To*>);

      static_assert(!std::derived_from<volatile From&&, To*>);
      static_assert(!std::derived_from<volatile From&&, const To*>);
      static_assert(!std::derived_from<volatile From&&, volatile To*>);
      static_assert(!std::derived_from<volatile From&&, const volatile To*>);

      static_assert(!std::derived_from<const volatile From&, To*>);
      static_assert(!std::derived_from<const volatile From&, const To*>);
      static_assert(!std::derived_from<const volatile From&, volatile To*>);
      static_assert(
          !std::derived_from<const volatile From&, const volatile To*>);

      static_assert(!std::derived_from<const volatile From&&, To*>);
      static_assert(!std::derived_from<const volatile From&&, const To*>);
      static_assert(!std::derived_from<const volatile From&&, volatile To*>);
      static_assert(
          !std::derived_from<const volatile From&&, const volatile To*>);
    }
  }
  { // Both as pointers
    static_assert(!std::derived_from<From*, To*>);
    static_assert(!std::derived_from<From*, const To*>);
    static_assert(!std::derived_from<From*, volatile To*>);
    static_assert(!std::derived_from<From*, const volatile To*>);

    static_assert(!std::derived_from<const From*, To*>);
    static_assert(!std::derived_from<const From*, const To*>);
    static_assert(!std::derived_from<const From*, volatile To*>);
    static_assert(!std::derived_from<const From*, const volatile To*>);

    static_assert(!std::derived_from<volatile From*, To*>);
    static_assert(!std::derived_from<volatile From*, const To*>);
    static_assert(!std::derived_from<volatile From*, volatile To*>);
    static_assert(!std::derived_from<volatile From*, const volatile To*>);

    static_assert(!std::derived_from<const volatile From*, To*>);
    static_assert(!std::derived_from<const volatile From*, const To*>);
    static_assert(!std::derived_from<const volatile From*, volatile To*>);
    static_assert(!std::derived_from<const volatile From*, const volatile To*>);
  }

  // From as the return type of a pointer-to-function
  if constexpr (!std::is_array_v<From>) {
    static_assert(!std::derived_from<From (*)(), To>);
    static_assert(!std::derived_from<From (*)(int), To>);
  }

  // To as the return type of a pointer-to-function
  if constexpr (!std::is_array_v<To>) {
    static_assert(!std::derived_from<From, To (*)()>);
    static_assert(!std::derived_from<From, To (*)(double)>);
  }

  // Both as the return type of a pointer-to-function
  if constexpr (!std::is_array_v<From> && !std::is_array_v<To>) {
    static_assert(!std::derived_from<From (*)(), To (*)()>);
    static_assert(!std::derived_from<From (*)(int), To (*)(double)>);
  }
  { // pointer-to-member
    if constexpr (std::is_class_v<From> && !std::same_as<To, void>) {
      static_assert(!std::derived_from<To From::*, To>);
    }

    if constexpr (std::is_class_v<To> && !std::same_as<From, void>) {
      static_assert(!std::derived_from<From To::*, From>);
    }
  }
  { // pointer-to-member-functions
    if constexpr (std::is_class_v<From>) {
      static_assert(!std::derived_from<From (From::*)(), To>);
    }

    if constexpr (std::is_class_v<To>) {
      static_assert(!std::derived_from<To (To::*)(), From>);
    }
  }
}

template <typename From, typename To>
constexpr void CheckNotDerivedFromReference() {
  if constexpr (!std::same_as<To, void>) {
    static_assert(!std::derived_from<From, To&>);
    static_assert(!std::derived_from<From, const To&>);
    static_assert(!std::derived_from<From, volatile To&>);
    static_assert(!std::derived_from<From, const volatile To&>);

    static_assert(!std::derived_from<From, To&&>);
    static_assert(!std::derived_from<From, const To&&>);
    static_assert(!std::derived_from<From, volatile To&&>);
    static_assert(!std::derived_from<From, const volatile To&&>);
  }

  if constexpr (!std::same_as<From, void>) {
    static_assert(!std::derived_from<From&, To>);
    static_assert(!std::derived_from<From&, To>);
    static_assert(!std::derived_from<From&, To>);
    static_assert(!std::derived_from<From&, To>);

    static_assert(!std::derived_from<From&&, To>);
    static_assert(!std::derived_from<From&&, To>);
    static_assert(!std::derived_from<From&&, To>);
    static_assert(!std::derived_from<From&&, To>);
  }

  // From as lvalue references
  if constexpr (!std::same_as<From, void> && !std::same_as<To, void>) {
    static_assert(!std::derived_from<From&, To&>);
    static_assert(!std::derived_from<From&, const To&>);
    static_assert(!std::derived_from<From&, volatile To&>);
    static_assert(!std::derived_from<From&, const volatile To&>);

    static_assert(!std::derived_from<From&, To&&>);
    static_assert(!std::derived_from<From&, const To&&>);
    static_assert(!std::derived_from<From&, volatile To&&>);
    static_assert(!std::derived_from<From&, const volatile To&&>);

    static_assert(!std::derived_from<const From&, To&>);
    static_assert(!std::derived_from<const From&, const To&>);
    static_assert(!std::derived_from<const From&, volatile To&>);
    static_assert(!std::derived_from<const From&, const volatile To&>);

    static_assert(!std::derived_from<const From&, To&&>);
    static_assert(!std::derived_from<const From&, const To&&>);
    static_assert(!std::derived_from<const From&, volatile To&&>);
    static_assert(!std::derived_from<const From&, const volatile To&&>);

    static_assert(!std::derived_from<volatile From&, To&>);
    static_assert(!std::derived_from<volatile From&, const To&>);
    static_assert(!std::derived_from<volatile From&, volatile To&>);
    static_assert(!std::derived_from<volatile From&, const volatile To&>);

    static_assert(!std::derived_from<volatile From&, To&&>);
    static_assert(!std::derived_from<volatile From&, const To&&>);
    static_assert(!std::derived_from<volatile From&, volatile To&&>);
    static_assert(!std::derived_from<volatile From&, const volatile To&&>);

    static_assert(!std::derived_from<const volatile From&, To&>);
    static_assert(!std::derived_from<const volatile From&, const To&>);
    static_assert(!std::derived_from<const volatile From&, volatile To&>);
    static_assert(!std::derived_from<const volatile From&, const volatile To&>);

    static_assert(!std::derived_from<const volatile From&, To&&>);
    static_assert(!std::derived_from<const volatile From&, const To&&>);
    static_assert(!std::derived_from<const volatile From&, volatile To&&>);
    static_assert(
        !std::derived_from<const volatile From&, const volatile To&&>);

    // From as rvalue references
    static_assert(!std::derived_from<From&&, To&>);
    static_assert(!std::derived_from<From&&, const To&>);
    static_assert(!std::derived_from<From&&, volatile To&>);
    static_assert(!std::derived_from<From&&, const volatile To&>);

    static_assert(!std::derived_from<From&&, To&&>);
    static_assert(!std::derived_from<From&&, const To&&>);
    static_assert(!std::derived_from<From&&, volatile To&&>);
    static_assert(!std::derived_from<From&&, const volatile To&&>);

    static_assert(!std::derived_from<const From&&, To&>);
    static_assert(!std::derived_from<const From&&, const To&>);
    static_assert(!std::derived_from<const From&&, volatile To&>);
    static_assert(!std::derived_from<const From&&, const volatile To&>);

    static_assert(!std::derived_from<const From&&, To&&>);
    static_assert(!std::derived_from<const From&&, const To&&>);
    static_assert(!std::derived_from<const From&&, volatile To&&>);
    static_assert(!std::derived_from<const From&&, const volatile To&&>);

    static_assert(!std::derived_from<volatile From&&, To&>);
    static_assert(!std::derived_from<volatile From&&, const To&>);
    static_assert(!std::derived_from<volatile From&&, volatile To&>);
    static_assert(!std::derived_from<volatile From&&, const volatile To&>);

    static_assert(!std::derived_from<volatile From&&, To&&>);
    static_assert(!std::derived_from<volatile From&&, const To&&>);
    static_assert(!std::derived_from<volatile From&&, volatile To&&>);
    static_assert(!std::derived_from<volatile From&&, const volatile To&&>);

    static_assert(!std::derived_from<const volatile From&&, To&>);
    static_assert(!std::derived_from<const volatile From&&, const To&>);
    static_assert(!std::derived_from<const volatile From&&, volatile To&>);
    static_assert(
        !std::derived_from<const volatile From&&, const volatile To&>);

    static_assert(!std::derived_from<const volatile From&&, To&&>);
    static_assert(!std::derived_from<const volatile From&&, const To&&>);
    static_assert(!std::derived_from<const volatile From&&, volatile To&&>);
    static_assert(
        !std::derived_from<const volatile From&&, const volatile To&&>);
  }

  // From as the return type of a reference-to-function
  if constexpr (!std::is_array_v<From>) {
    static_assert(!std::derived_from<From (&)(), To>);
    static_assert(!std::derived_from<From (&)(int), To>);
  }
  // To as the return type of a reference-to-function
  if constexpr (!std::is_array_v<To>) {
    static_assert(!std::derived_from<From, To (&)()>);
    static_assert(!std::derived_from<From, To (&)(double)>);
  }
  // Both as the return type of a reference-to-function
  if constexpr (!std::is_array_v<From> && !std::is_array_v<To>) {
    static_assert(!std::derived_from<From (&)(), To (&)()>);
    static_assert(!std::derived_from<From (&)(int), To (&)(double)>);
  }
}

template <typename From, typename To>
constexpr void CheckDerivedFrom() {
  static_assert(std::derived_from<From, To>);

  static_assert(std::derived_from<From, const To>);
  static_assert(std::derived_from<From, volatile To>);
  static_assert(std::derived_from<From, const volatile To>);

  static_assert(std::derived_from<const From, const To>);
  static_assert(std::derived_from<const From, volatile To>);
  static_assert(std::derived_from<const From, const volatile To>);

  static_assert(std::derived_from<volatile From, const To>);
  static_assert(std::derived_from<volatile From, volatile To>);
  static_assert(std::derived_from<volatile From, const volatile To>);

  static_assert(std::derived_from<const volatile From, const To>);
  static_assert(std::derived_from<const volatile From, volatile To>);
  static_assert(std::derived_from<const volatile From, const volatile To>);

  CheckNotDerivedFromPointer<From, To>();
  CheckNotDerivedFromReference<From, To>();
}

template <typename From, typename To>
constexpr void CheckNotDerivedFrom() {
  static_assert(!std::derived_from<From, To>);

  static_assert(!std::derived_from<From, const To>);
  static_assert(!std::derived_from<From, volatile To>);
  static_assert(!std::derived_from<From, const volatile To>);

  static_assert(!std::derived_from<const From, const To>);
  static_assert(!std::derived_from<const From, volatile To>);
  static_assert(!std::derived_from<const From, const volatile To>);

  static_assert(!std::derived_from<volatile From, const To>);
  static_assert(!std::derived_from<volatile From, volatile To>);
  static_assert(!std::derived_from<volatile From, const volatile To>);

  static_assert(!std::derived_from<const volatile From, const To>);
  static_assert(!std::derived_from<const volatile From, volatile To>);
  static_assert(!std::derived_from<const volatile From, const volatile To>);

  CheckNotDerivedFromPointer<From, To>();
  CheckNotDerivedFromReference<From, To>();
}

enum Enumeration { Yes, No };
enum class ScopedEnumeration : int { No, Yes };

int main(int, char**) {
  { // Fundamentals shouldn't be derived from anything
    CheckNotDerivedFrom<int, long>();
    CheckNotDerivedFrom<signed char, char>();
    CheckNotDerivedFrom<double, Base1>();

    CheckNotDerivedFrom<int, Enumeration>();
    CheckNotDerivedFrom<int, ScopedEnumeration>();

    CheckNotDerivedFrom<void, void>();
    CheckNotDerivedFrom<int, int>();
  }
  { // Nothing should be derived from a fundamental type
    CheckNotDerivedFrom<Enumeration, int>();
    CheckNotDerivedFrom<ScopedEnumeration, int>();

    CheckNotDerivedFrom<Base1, int>();
    CheckNotDerivedFrom<Base1, double>();
    CheckNotDerivedFrom<Derived1, char>();
    CheckNotDerivedFrom<DerivedPrivate, long long>();
  }
  { // Other built-in things shouldn't have derivations
    CheckNotDerivedFrom<Enumeration, Enumeration>();
    CheckNotDerivedFrom<ScopedEnumeration, ScopedEnumeration>();

    CheckNotDerivedFrom<Enumeration, ScopedEnumeration>();
    CheckNotDerivedFrom<ScopedEnumeration, Enumeration>();

    CheckNotDerivedFrom<Base1[5], Base1>();
    CheckNotDerivedFrom<Derived1[5], Base1>();

    CheckNotDerivedFrom<Base1, Base1[5]>();
    CheckNotDerivedFrom<Derived1, Base1[5]>();
  }

  { // Base1 is the subject.
    CheckDerivedFrom<Base1, Base1>();

    CheckNotDerivedFrom<Base1, void>();
    CheckNotDerivedFrom<Base1, DerivedPrivate>();
    CheckNotDerivedFrom<Base1, DerivedProtected>();
    CheckNotDerivedFrom<Base1, Derived1>();
    CheckNotDerivedFrom<Base1, Derived2>();
    CheckNotDerivedFrom<Base1, Derived3>();
    CheckNotDerivedFrom<Base1, Derived4>();
  }

  { // Derived1 is the subject.
    CheckDerivedFrom<Derived1, Base1>();
    CheckDerivedFrom<Derived1, Derived1>();

    CheckNotDerivedFromPointer<Derived1, void>();
    CheckNotDerivedFrom<Derived1, DerivedPrivate>();
    CheckNotDerivedFrom<Derived1, DerivedProtected>();
    CheckNotDerivedFrom<Derived1, Derived2>();
    CheckNotDerivedFrom<Derived1, Derived3>();
    CheckNotDerivedFrom<Derived1, Derived4>();
  }

  { // Derived2 is the subject.
    CheckDerivedFrom<Derived2, Base1>();
    CheckDerivedFrom<Derived2, Derived2>();

    CheckNotDerivedFrom<Derived2, DerivedPrivate>();
    CheckNotDerivedFrom<Derived2, DerivedProtected>();
    CheckNotDerivedFrom<Derived2, Derived1>();
    CheckNotDerivedFrom<Derived2, Derived3>();
    CheckNotDerivedFrom<Derived2, Derived4>();
  }

  { // DerivedPrivate is the subject.
    CheckDerivedFrom<DerivedPrivate, DerivedPrivate>();

    CheckNotDerivedFrom<DerivedPrivate, Base1>();
    CheckNotDerivedFrom<DerivedPrivate, DerivedProtected>();
    CheckNotDerivedFrom<DerivedPrivate, Derived1>();
    CheckNotDerivedFrom<DerivedPrivate, Derived2>();
    CheckNotDerivedFrom<DerivedPrivate, Derived3>();
    CheckNotDerivedFrom<DerivedPrivate, Derived4>();
  }

  { // Derived3 is the subject.
    CheckDerivedFrom<Derived3, DerivedPrivate>();
    CheckDerivedFrom<Derived3, Derived3>();

    CheckNotDerivedFrom<Derived3, Base1>();
    CheckNotDerivedFrom<Derived3, DerivedProtected>();
    CheckNotDerivedFrom<Derived3, Derived1>();
    CheckNotDerivedFrom<Derived3, Derived2>();
    CheckNotDerivedFrom<Derived3, Derived4>();
  }

  { // DerivedProtected is the subject.
    CheckDerivedFrom<DerivedProtected, DerivedProtected>();

    CheckNotDerivedFromPointer<DerivedProtected, Base1>();
    CheckNotDerivedFromPointer<DerivedProtected, DerivedPrivate>();
    CheckNotDerivedFromPointer<DerivedProtected, Derived1>();
    CheckNotDerivedFromPointer<DerivedProtected, Derived2>();
    CheckNotDerivedFromPointer<DerivedProtected, Derived3>();
    CheckNotDerivedFromPointer<DerivedProtected, Derived4>();
  }

  { // Derived4 is the subject.
    CheckDerivedFrom<Derived4, DerivedProtected>();
    CheckDerivedFrom<Derived4, Derived4>();

    CheckNotDerivedFrom<Derived4, Base1>();
    CheckNotDerivedFrom<Derived4, DerivedPrivate>();
    CheckNotDerivedFrom<Derived4, Derived1>();
    CheckNotDerivedFrom<Derived4, Derived2>();
    CheckNotDerivedFrom<Derived4, Derived3>();
  }

  { // Derived5 is the subject.
    CheckDerivedFrom<Derived5, DerivedProtected>();
    CheckDerivedFrom<Derived5, Derived4>();
    CheckDerivedFrom<Derived5, Derived5>();

    CheckNotDerivedFrom<Derived5, Base1>();
    CheckNotDerivedFrom<Derived5, DerivedPrivate>();
    CheckNotDerivedFrom<Derived5, Derived1>();
    CheckNotDerivedFrom<Derived5, Derived2>();
    CheckNotDerivedFrom<Derived5, Derived3>();
  }

  return 0;
}
