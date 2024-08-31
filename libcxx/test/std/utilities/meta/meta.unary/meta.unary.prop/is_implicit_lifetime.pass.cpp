//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// These compilers don't support __builtin_is_implicit_lifetime yet.
// UNSUPPORTED: clang-17, clang-18, clang-19, gcc-14, apple-clang-16, apple-clang-17
// XFAIL: apple-clang

// <type_traits>

// template<class T> struct is_implicit_lifetime;

#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_macros.h"

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

#ifdef TEST_COMPILER_CLANG
using AlignValueInt = int* __attribute__((align_value(16)));
using Float4        = float __attribute__((ext_vector_type(4)));

struct [[clang::enforce_read_only_placement]] EnforceReadOnlyPlacement {};
struct [[clang::type_visibility("hidden")]] TypeVisibility {};
#endif

// Test implicit-lifetime type
template <typename T, bool Expected>
constexpr void test_is_implicit_lifetime() {
  static_assert(std::is_implicit_lifetime<T>::value == Expected);

  // Runtime check
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
  test_is_implicit_lifetime<T* __restrict, true>();

  // Arrays
  test_is_implicit_lifetime<T[], true>();
  test_is_implicit_lifetime<T[0], true>();
  test_is_implicit_lifetime<T[94], true>();
}

constexpr bool test() {
  // Standard fundamental C++ types

  test_is_implicit_lifetime<decltype(nullptr), true>();
  test_is_implicit_lifetime<std::nullptr_t, true>();

  test_is_implicit_lifetime<void, false>();
  test_is_implicit_lifetime<const void, false>();
  test_is_implicit_lifetime<volatile void, false>();

  test_is_implicit_lifetime<bool>();

  test_is_implicit_lifetime<char>();
  test_is_implicit_lifetime<signed char>();
  test_is_implicit_lifetime<unsigned char>();

#if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  test_is_implicit_lifetime<wchar_t>();
#endif

#if !defined(TEST_HAS_NO_CHAR8_T)
  test_is_implicit_lifetime<char8_t>();
#endif
  test_is_implicit_lifetime<char16_t>();
  test_is_implicit_lifetime<char32_t>();

  test_is_implicit_lifetime<short>();
  test_is_implicit_lifetime<short int>();
  test_is_implicit_lifetime<signed short>();
  test_is_implicit_lifetime<signed short int>();
  test_is_implicit_lifetime<unsigned short>();
  test_is_implicit_lifetime<unsigned short int>();
  test_is_implicit_lifetime<int>();
  test_is_implicit_lifetime<signed>();
  test_is_implicit_lifetime<signed int>();
  test_is_implicit_lifetime<unsigned>();
  test_is_implicit_lifetime<unsigned int>();
  test_is_implicit_lifetime<long>();
  test_is_implicit_lifetime<long int>();
  test_is_implicit_lifetime<signed long>();
  test_is_implicit_lifetime<signed long int>();
  test_is_implicit_lifetime<unsigned long>();
  test_is_implicit_lifetime<unsigned long int>();
  test_is_implicit_lifetime<long long>();
  test_is_implicit_lifetime<long long int>();
  test_is_implicit_lifetime<signed long long>();
  test_is_implicit_lifetime<signed long long int>();
  test_is_implicit_lifetime<unsigned long long>();
  test_is_implicit_lifetime<unsigned long long int>();

  test_is_implicit_lifetime<float>();
  test_is_implicit_lifetime<double>();
  test_is_implicit_lifetime<long double>();

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

  // Standard C types

  test_is_implicit_lifetime<_Complex float>();
  test_is_implicit_lifetime<_Complex double>();
  test_is_implicit_lifetime<_Complex long double>();

  // Standard C23 types

#ifdef TEST_COMPILER_CLANG
  test_is_implicit_lifetime<_BitInt(8)>();
  test_is_implicit_lifetime<_BitInt(128)>();
#endif

  // Language extensions: Types

#if !defined(TEST_HAS_NO_INT128)
  test_is_implicit_lifetime<__int128_t>();
  test_is_implicit_lifetime<__uint128_t>();
#endif

#ifdef TEST_COMPILER_CLANG
  // https://clang.llvm.org/docs/LanguageExtensions.html#half-precision-floating-point
  test_is_implicit_lifetime<__fp16>();
  test_is_implicit_lifetime<__bf16>();
#endif // TEST_COMPILER_CLANG

  // Language extensions: Attributes

#ifdef TEST_COMPILER_CLANG
  test_is_implicit_lifetime<AlignValueInt, true>();
  test_is_implicit_lifetime<Float4, true>();

  test_is_implicit_lifetime<EnforceReadOnlyPlacement, true>();
  test_is_implicit_lifetime<TypeVisibility, true>();

  test_is_implicit_lifetime<int [[clang::annotate_type("category2")]]*, true>();
  test_is_implicit_lifetime<int __attribute__((btf_type_tag("user")))*, true>();

  test_is_implicit_lifetime<int __attribute__((noderef))*, true>();

  test_is_implicit_lifetime<int* _Nonnull, true>();
  test_is_implicit_lifetime<int* _Null_unspecified, true>();
  test_is_implicit_lifetime<int* _Nullable, true>();
#endif // TEST_COMPILER_CLANG

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
