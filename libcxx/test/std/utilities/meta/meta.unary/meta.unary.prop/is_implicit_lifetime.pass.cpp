//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// These compilers don't support __builtin_is_implicit_lifetime yet.
// UNSUPPORTED: clang-18, clang-19, gcc-14, apple-clang-15, apple-clang-16, apple-clang-17

// <type_traits>

// template<class T> struct is_implicit_lifetime;

#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "type_algorithms.h"

enum Enum { EV };
enum SignedEnum : signed int {};
enum UnsignedEnum : unsigned int {};

enum class EnumClass { EV };
enum class SignedEnumClass : signed int {};
enum class UnsignedEnumClass : unsigned int {};

struct EmptyStruct {};
struct IncompleteStruct;

struct NoEligibleTrivialContructor {
  NoEligibleTrivialContructor() {};
  NoEligibleTrivialContructor(const NoEligibleTrivialContructor&) {}
  NoEligibleTrivialContructor(NoEligibleTrivialContructor&&) {}
};

struct OnlyDefaultConstructorIsTrivial {
  OnlyDefaultConstructorIsTrivial() = default;
  OnlyDefaultConstructorIsTrivial(const OnlyDefaultConstructorIsTrivial&) {}
  OnlyDefaultConstructorIsTrivial(OnlyDefaultConstructorIsTrivial&&) {}
};

struct AllContstructorsAreTrivial {
  AllContstructorsAreTrivial()                                  = default;
  AllContstructorsAreTrivial(const AllContstructorsAreTrivial&) = default;
  AllContstructorsAreTrivial(AllContstructorsAreTrivial&&)      = default;
};

struct InheritedNoEligibleTrivialConstructor : NoEligibleTrivialContructor {
  using NoEligibleTrivialContructor::NoEligibleTrivialContructor;
};

struct InheritedOnlyDefaultConstructorIsTrivial : OnlyDefaultConstructorIsTrivial {
  using OnlyDefaultConstructorIsTrivial::OnlyDefaultConstructorIsTrivial;
};

struct InheritedAllContstructorsAreTrivial : AllContstructorsAreTrivial {
  using AllContstructorsAreTrivial::AllContstructorsAreTrivial;
};

struct UserDeclaredDestructor {
  ~UserDeclaredDestructor() = default;
};

struct UserProvidedDestructor {
  ~UserProvidedDestructor() {}
};

struct UserDeletedDestructorInAggregate {
  ~UserDeletedDestructorInAggregate() = delete;
};

struct UserDeletedDestructorInNonAggregate {
  virtual void NonAggregate();
  ~UserDeletedDestructorInNonAggregate() = delete;
};

struct DeletedDestructorViaBaseInAggregate : UserDeletedDestructorInAggregate {};
struct DeletedDestructorViaBaseInNonAggregate : UserDeletedDestructorInNonAggregate {};

template <bool B>
struct ConstrainedUserDeclaredDefaultConstructor {
  ConstrainedUserDeclaredDefaultConstructor()
    requires B
  = default;
  ConstrainedUserDeclaredDefaultConstructor(const ConstrainedUserDeclaredDefaultConstructor&) {}
};

template <bool B>
struct ConstrainedUserProvidedDestructor {
  ~ConstrainedUserProvidedDestructor() = default;
  ~ConstrainedUserProvidedDestructor()
    requires B
  {}
};

struct StructWithFlexibleArrayMember {
  int arr[];
};

struct StructWithZeroSizedArray {
  int arr[0];
};

// Test implicit-lifetime type
template <typename T, bool Expected>
constexpr void test_is_implicit_lifetime() {
  assert(std::is_implicit_lifetime<T>::value == Expected);
  assert(std::is_implicit_lifetime_v<T> == Expected);
}

// Test pointer, reference, array, etc. types
template <typename T>
constexpr void test_is_implicit_lifetime() {
  test_is_implicit_lifetime<T, true>();

  // cv-qualified
  test_is_implicit_lifetime<const T, true>();
  test_is_implicit_lifetime<volatile T, true>();

  test_is_implicit_lifetime<T&, false>();
  test_is_implicit_lifetime<T&&, false>();

  // Pointer types
  test_is_implicit_lifetime<T*, true>();

  // Arrays
  test_is_implicit_lifetime<T[], true>();
  test_is_implicit_lifetime<T[94], true>();
}

struct AritmeticTypesTest {
  template <class T>
  constexpr void operator()() {
    test_is_implicit_lifetime<T>();
  }
};

constexpr bool test() {
  // Standard fundamental C++ types

  test_is_implicit_lifetime<std::nullptr_t, true>();

  test_is_implicit_lifetime<void, false>();
  test_is_implicit_lifetime<const void, false>();
  test_is_implicit_lifetime<volatile void, false>();

  types::for_each(types::arithmetic_types(), AritmeticTypesTest{});

  test_is_implicit_lifetime<Enum>();
  test_is_implicit_lifetime<SignedEnum>();
  test_is_implicit_lifetime<UnsignedEnum>();

  test_is_implicit_lifetime<EnumClass>();
  test_is_implicit_lifetime<SignedEnumClass>();
  test_is_implicit_lifetime<UnsignedEnumClass>();

  test_is_implicit_lifetime<void(), false>();
  test_is_implicit_lifetime<void()&, false>();
  test_is_implicit_lifetime<void() const, false>();
  test_is_implicit_lifetime<void (&)(), false>();
  test_is_implicit_lifetime<void (*)(), true>();

  // Implicit-lifetime class types

  test_is_implicit_lifetime<EmptyStruct>();
  test_is_implicit_lifetime<int EmptyStruct::*, true>(); // Pointer-to-member
  test_is_implicit_lifetime<int (EmptyStruct::*)(), true>();
  test_is_implicit_lifetime<int (EmptyStruct::*)() const, true>();
  test_is_implicit_lifetime<int (EmptyStruct::*)()&, true>();
  test_is_implicit_lifetime<int (EmptyStruct::*)()&&, true>();

  test_is_implicit_lifetime<IncompleteStruct[], true>();
  test_is_implicit_lifetime<IncompleteStruct[82], true>();

  test_is_implicit_lifetime<UserDeclaredDestructor>();

  test_is_implicit_lifetime<UserProvidedDestructor, false>();

  test_is_implicit_lifetime<NoEligibleTrivialContructor, false>();

  test_is_implicit_lifetime<OnlyDefaultConstructorIsTrivial, true>();

  test_is_implicit_lifetime<AllContstructorsAreTrivial, true>();

  test_is_implicit_lifetime<InheritedNoEligibleTrivialConstructor, false>();

  test_is_implicit_lifetime<InheritedOnlyDefaultConstructorIsTrivial, true>();

  test_is_implicit_lifetime<InheritedAllContstructorsAreTrivial, true>();

  test_is_implicit_lifetime<UserDeletedDestructorInAggregate, true>();

  test_is_implicit_lifetime<UserDeletedDestructorInNonAggregate, false>();

  test_is_implicit_lifetime<DeletedDestructorViaBaseInAggregate, true>();

  test_is_implicit_lifetime<DeletedDestructorViaBaseInNonAggregate, false>();

  test_is_implicit_lifetime<ConstrainedUserDeclaredDefaultConstructor<true>, true>();
  test_is_implicit_lifetime<ConstrainedUserDeclaredDefaultConstructor<false>, false>();

  test_is_implicit_lifetime<ConstrainedUserProvidedDestructor<true>, false>();
  test_is_implicit_lifetime<ConstrainedUserProvidedDestructor<false>, true>();

  test_is_implicit_lifetime<StructWithFlexibleArrayMember, true>();

  test_is_implicit_lifetime<StructWithZeroSizedArray, true>();

  // C++ standard library types

  test_is_implicit_lifetime<std::pair<int, float>>();
  test_is_implicit_lifetime<std::tuple<int, float>>();

  // Standard C23 types

#ifdef TEST_COMPILER_CLANG
  test_is_implicit_lifetime<_BitInt(8)>();
  test_is_implicit_lifetime<_BitInt(128)>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
