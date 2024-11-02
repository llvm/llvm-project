// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_POISONED_HASH_HELPER_H
#define SUPPORT_POISONED_HASH_HELPER_H

#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "test_workarounds.h"

#if TEST_STD_VER < 11
#error this header may only be used in C++11 or newer
#endif

template <class ...Args> struct TypeList;

// Test that the specified Hash meets the requirements of an enabled hash
template <class Hash, class Key, class InputKey = Key>
TEST_CONSTEXPR_CXX20 void test_hash_enabled(InputKey const& key = InputKey{});

template <class T, class InputKey = T>
TEST_CONSTEXPR_CXX20 void test_hash_enabled_for_type(InputKey const& key = InputKey{}) {
  return test_hash_enabled<std::hash<T>, T, InputKey>(key);
}

// Test that the specified Hash meets the requirements of a disabled hash.
template <class Hash, class Key>
void test_hash_disabled();

template <class T>
void test_hash_disabled_for_type() {
  return test_hash_disabled<std::hash<T>, T>();
}

namespace PoisonedHashDetail {
  enum Enum {};
  enum EnumClass : bool {};
  struct Class {};
}

// Each header that declares the template hash provides enabled
// specializations of hash for nullptr t and all cv-unqualified
// arithmetic, enumeration, and pointer types.
using LibraryHashTypes = TypeList<
#if TEST_STD_VER > 14
      decltype(nullptr),
#endif
      bool,
      char,
      signed char,
      unsigned char,
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
      wchar_t,
#endif
      char16_t,
      char32_t,
      short,
      unsigned short,
      int,
      unsigned int,
      long,
      unsigned long,
      long long,
      unsigned long long,
#ifndef TEST_HAS_NO_INT128
      __int128_t,
      __uint128_t,
#endif
      float,
      double,
      long double,
      PoisonedHashDetail::Enum,
      PoisonedHashDetail::EnumClass,
      void*,
      void const*,
      PoisonedHashDetail::Class*
    >;


// Test that each of the library hash specializations for  arithmetic types,
// enum types, and pointer types are available and enabled.
template <class Types = LibraryHashTypes>
void test_library_hash_specializations_available(Types = Types{});


namespace PoisonedHashDetail {

template <class T, class = typename T::foo_bar_baz>
constexpr bool instantiate(int) { return true; }
template <class> constexpr bool instantiate(long) { return true; }
template <class T> constexpr bool instantiate() { return instantiate<T>(0); }

template <class To>
struct ConvertibleToSimple {
  operator To() const {
    return To{};
  }
};

template <class To>
struct ConvertibleTo {
  To to{};
  operator To&() & { return to; }
  operator To const&() const & { return to; }
  operator To&&() && { return std::move(to); }
  operator To const&&() const && { return std::move(to); }
};

template <class Hasher, class Key, class Res = decltype(std::declval<Hasher&>()(std::declval<Key>()))>
constexpr bool can_hash(int) {
  return std::is_same<Res, size_t>::value;
}
template <class, class>
constexpr bool can_hash(long) {
  return false;
}
template <class Hasher, class Key>
constexpr bool can_hash() {
  return can_hash<Hasher, Key>(0);
}
} // namespace PoisonedHashDetail

template <class Hash, class Key, class InputKey>
TEST_CONSTEXPR_CXX20 void test_hash_enabled(InputKey const& key) {
  using namespace PoisonedHashDetail;

  static_assert(std::is_destructible<Hash>::value, "");
  // Enabled hash requirements
  static_assert(std::is_default_constructible<Hash>::value, "");
  static_assert(std::is_copy_constructible<Hash>::value, "");
  static_assert(std::is_move_constructible<Hash>::value, "");
  static_assert(std::is_copy_assignable<Hash>::value, "");
  static_assert(std::is_move_assignable<Hash>::value, "");

#if TEST_STD_VER > 14
  static_assert(std::is_swappable<Hash>::value, "");
#elif defined(_LIBCPP_VERSION)
  static_assert(std::__is_swappable<Hash>::value, "");
#endif

  // Hashable requirements
  static_assert(can_hash<Hash, Key&>(), "");
  static_assert(can_hash<Hash, Key const&>(), "");
  static_assert(can_hash<Hash, Key&&>(), "");
  static_assert(can_hash<Hash const, Key&>(), "");
  static_assert(can_hash<Hash const, Key const&>(), "");
  static_assert(can_hash<Hash const, Key&&>(), "");

  static_assert(can_hash<Hash, ConvertibleToSimple<Key>&>(), "");
  static_assert(can_hash<Hash, ConvertibleToSimple<Key> const&>(), "");
  static_assert(can_hash<Hash, ConvertibleToSimple<Key>&&>(), "");

  static_assert(can_hash<Hash, ConvertibleTo<Key>&>(), "");
  static_assert(can_hash<Hash, ConvertibleTo<Key> const&>(), "");
  static_assert(can_hash<Hash, ConvertibleTo<Key>&&>(), "");
  static_assert(can_hash<Hash, ConvertibleTo<Key> const&&>(), "");

  const Hash h{};
  assert(h(key) == h(key));

}

template <class Hash, class Key>
void test_hash_disabled() {
  using namespace PoisonedHashDetail;

  // Disabled hash requirements
  static_assert(!std::is_default_constructible<Hash>::value, "");
  static_assert(!std::is_copy_constructible<Hash>::value, "");
  static_assert(!std::is_move_constructible<Hash>::value, "");
  static_assert(!std::is_copy_assignable<Hash>::value, "");
  static_assert(!std::is_move_assignable<Hash>::value, "");

  static_assert(!std::is_function<
      typename std::remove_pointer<
          typename std::remove_reference<Hash>::type
      >::type
    >::value, "");

  // Hashable requirements
  static_assert(!can_hash<Hash, Key&>(), "");
  static_assert(!can_hash<Hash, Key const&>(), "");
  static_assert(!can_hash<Hash, Key&&>(), "");
  static_assert(!can_hash<Hash const, Key&>(), "");
  static_assert(!can_hash<Hash const, Key const&>(), "");
  static_assert(!can_hash<Hash const, Key&&>(), "");

  static_assert(!can_hash<Hash, ConvertibleToSimple<Key>&>(), "");
  static_assert(!can_hash<Hash, ConvertibleToSimple<Key> const&>(), "");
  static_assert(!can_hash<Hash, ConvertibleToSimple<Key>&&>(), "");

  static_assert(!can_hash<Hash, ConvertibleTo<Key>&>(), "");
  static_assert(!can_hash<Hash, ConvertibleTo<Key> const&>(), "");
  static_assert(!can_hash<Hash, ConvertibleTo<Key>&&>(), "");
  static_assert(!can_hash<Hash, ConvertibleTo<Key> const&&>(), "");
}


template <class First, class ...Rest>
struct TypeList<First, Rest...> {
  template <template <class> class Trait, bool Expect = true>
  static constexpr bool assertTrait() {
    static_assert(Trait<First>::value == Expect, "");
    return TypeList<Rest...>::template assertTrait<Trait, Expect>();
  }

  template <class Trait>
  static void applyTrait() {
    Trait::template apply<First>();
    TypeList<Rest...>::template applyTrait<Trait>();
  }
};

template <>
struct TypeList<> {
  template <template <class> class Trait, bool Expect = true>
  static constexpr bool assertTrait() {
    return true;
  }
  template <class Trait>
  static void applyTrait() {}
};


struct TestLibraryTrait {
    template <class Type>
    static void apply() { test_hash_enabled<std::hash<Type>, Type>(); }
};

template <class Types>
void test_library_hash_specializations_available(Types) {
  Types::template applyTrait<TestLibraryTrait >();
}

#endif // SUPPORT_POISONED_HASH_HELPER_H
