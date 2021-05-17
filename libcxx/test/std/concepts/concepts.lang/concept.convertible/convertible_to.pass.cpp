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
// concept convertible_to;

#include <concepts>
#include <type_traits>

namespace {
enum ClassicEnum { a, b };
enum class ScopedEnum { x, y };
struct Empty {};
using nullptr_t = decltype(nullptr);

template <class T, class U>
void CheckConvertibleTo() {
  static_assert(std::convertible_to<T, U>);
  static_assert(std::convertible_to<const T, U>);
  static_assert(std::convertible_to<T, const U>);
  static_assert(std::convertible_to<const T, const U>);
}

template <class T, class U>
void CheckNotConvertibleTo() {
  static_assert(!std::convertible_to<T, U>);
  static_assert(!std::convertible_to<const T, U>);
  static_assert(!std::convertible_to<T, const U>);
  static_assert(!std::convertible_to<const T, const U>);
}

template <class T, class U>
void CheckIsConvertibleButNotConvertibleTo() {
  // Sanity check T is either implicitly xor explicitly convertible to U.
  static_assert(std::is_convertible_v<T, U>);
  static_assert(std::is_convertible_v<const T, U>);
  static_assert(std::is_convertible_v<T, const U>);
  static_assert(std::is_convertible_v<const T, const U>);
  CheckNotConvertibleTo<T, U>();
}

// Tests that should objectively return false (except for bool and nullptr_t)
template <class T>
constexpr void CommonlyNotConvertibleTo() {
  CheckNotConvertibleTo<T, void>();
  CheckNotConvertibleTo<T, nullptr_t>();
  CheckNotConvertibleTo<T, T*>();
  CheckNotConvertibleTo<T, T Empty::*>();
  CheckNotConvertibleTo<T, T (Empty::*)()>();
  CheckNotConvertibleTo<T, T[sizeof(T)]>();
  CheckNotConvertibleTo<T, T (*)()>();
  CheckNotConvertibleTo<T, T (&)()>();
  CheckNotConvertibleTo<T, T(&&)()>();
}

template <std::same_as<bool> >
constexpr void CommonlyNotConvertibleTo() {
  CheckNotConvertibleTo<bool, void>();
  CheckNotConvertibleTo<bool, nullptr_t>();
  CheckConvertibleTo<bool Empty::*, bool>();
  CheckConvertibleTo<bool (Empty::*)(), bool>();
  CheckConvertibleTo<bool[2], bool>();
  CheckConvertibleTo<bool (*)(), bool>();
  CheckConvertibleTo<bool (&)(), bool>();
  CheckConvertibleTo<bool(&&)(), bool>();
}

template <std::same_as<nullptr_t> >
constexpr void CommonlyNotConvertibleTo() {
  CheckNotConvertibleTo<nullptr_t, void>();
  CheckConvertibleTo<nullptr_t, nullptr_t>();
  CheckConvertibleTo<nullptr_t, void*>();
  CheckConvertibleTo<nullptr_t, int Empty::*>();
  CheckConvertibleTo<nullptr_t, void (Empty::*)()>();
  CheckNotConvertibleTo<nullptr_t, int[2]>();
  CheckConvertibleTo<nullptr_t, void (*)()>();
  CheckNotConvertibleTo<nullptr_t, void (&)()>();
  CheckNotConvertibleTo<nullptr_t, void(&&)()>();
}
} // namespace

using Function = void();
using NoexceptFunction = void() noexcept;
using ConstFunction = void() const;
using Array = char[1];

struct StringType {
  StringType(const char*) {}
};

class NonCopyable {
  NonCopyable(NonCopyable&);
};

template <typename T>
class CannotInstantiate {
  enum { X = T::ThisExpressionWillBlowUp };
};

struct abstract {
  virtual int f() = 0;
};

struct ExplicitlyConvertible;
struct ImplicitlyConvertible;

struct ExplicitlyConstructible {
  explicit ExplicitlyConstructible(int);
  explicit ExplicitlyConstructible(ExplicitlyConvertible);
  explicit ExplicitlyConstructible(ImplicitlyConvertible) = delete;
};

struct ExplicitlyConvertible {
  explicit operator ExplicitlyConstructible() const {
    return ExplicitlyConstructible(0);
  }
};

struct ImplicitlyConstructible;

struct ImplicitlyConvertible {
  operator ExplicitlyConstructible() const;
  operator ImplicitlyConstructible() const = delete;
};

struct ImplicitlyConstructible {
  ImplicitlyConstructible(ImplicitlyConvertible);
};

int main(int, char**) {
  // void
  CheckConvertibleTo<void, void>();
  CheckNotConvertibleTo<void, Function>();
  CheckNotConvertibleTo<void, Function&>();
  CheckNotConvertibleTo<void, Function*>();
  CheckNotConvertibleTo<void, NoexceptFunction>();
  CheckNotConvertibleTo<void, NoexceptFunction&>();
  CheckNotConvertibleTo<void, NoexceptFunction*>();
  CheckNotConvertibleTo<void, Array>();
  CheckNotConvertibleTo<void, Array&>();
  CheckNotConvertibleTo<void, char>();
  CheckNotConvertibleTo<void, char&>();
  CheckNotConvertibleTo<void, char*>();
  CheckNotConvertibleTo<char, void>();

  // Function
  CheckNotConvertibleTo<Function, void>();
  CheckNotConvertibleTo<Function, Function>();
  CheckNotConvertibleTo<Function, NoexceptFunction>();
  CheckNotConvertibleTo<Function, NoexceptFunction&>();
  CheckNotConvertibleTo<Function, NoexceptFunction*>();
  CheckNotConvertibleTo<Function, NoexceptFunction* const>();
  CheckConvertibleTo<Function, Function&>();
  CheckConvertibleTo<Function, Function*>();
  CheckConvertibleTo<Function, Function* const>();

  static_assert(std::convertible_to<Function, Function&&>);
  static_assert(!std::convertible_to<Function, NoexceptFunction&&>);

  CheckNotConvertibleTo<Function, Array>();
  CheckNotConvertibleTo<Function, Array&>();
  CheckNotConvertibleTo<Function, char>();
  CheckNotConvertibleTo<Function, char&>();
  CheckNotConvertibleTo<Function, char*>();

  // Function&
  CheckNotConvertibleTo<Function&, void>();
  CheckNotConvertibleTo<Function&, Function>();
  CheckConvertibleTo<Function&, Function&>();

  CheckConvertibleTo<Function&, Function*>();
  CheckNotConvertibleTo<Function&, Array>();
  CheckNotConvertibleTo<Function&, Array&>();
  CheckNotConvertibleTo<Function&, char>();
  CheckNotConvertibleTo<Function&, char&>();
  CheckNotConvertibleTo<Function&, char*>();

  // Function*
  CheckNotConvertibleTo<Function*, void>();
  CheckNotConvertibleTo<Function*, Function>();
  CheckNotConvertibleTo<Function*, Function&>();
  CheckConvertibleTo<Function*, Function*>();

  CheckNotConvertibleTo<Function*, Array>();
  CheckNotConvertibleTo<Function*, Array&>();
  CheckNotConvertibleTo<Function*, char>();
  CheckNotConvertibleTo<Function*, char&>();
  CheckNotConvertibleTo<Function*, char*>();

  // Non-referencable function type
  static_assert(!std::convertible_to<ConstFunction, Function>);
  static_assert(!std::convertible_to<ConstFunction, Function*>);
  static_assert(!std::convertible_to<ConstFunction, Function&>);
  static_assert(!std::convertible_to<ConstFunction, Function&&>);
  static_assert(!std::convertible_to<Function*, ConstFunction>);
  static_assert(!std::convertible_to<Function&, ConstFunction>);
  static_assert(!std::convertible_to<ConstFunction, ConstFunction>);
  static_assert(!std::convertible_to<ConstFunction, void>);

  // NoexceptFunction
  CheckNotConvertibleTo<NoexceptFunction, void>();
  CheckNotConvertibleTo<NoexceptFunction, Function>();
  CheckNotConvertibleTo<NoexceptFunction, NoexceptFunction>();
  CheckConvertibleTo<NoexceptFunction, NoexceptFunction&>();
  CheckConvertibleTo<NoexceptFunction, NoexceptFunction*>();
  CheckConvertibleTo<NoexceptFunction, NoexceptFunction* const>();
  CheckConvertibleTo<NoexceptFunction, Function&>();
  CheckConvertibleTo<NoexceptFunction, Function*>();
  CheckConvertibleTo<NoexceptFunction, Function* const>();

  static_assert(std::convertible_to<NoexceptFunction, Function&&>);
  static_assert(std::convertible_to<NoexceptFunction, NoexceptFunction&&>);

  CheckNotConvertibleTo<NoexceptFunction, Array>();
  CheckNotConvertibleTo<NoexceptFunction, Array&>();
  CheckNotConvertibleTo<NoexceptFunction, char>();
  CheckNotConvertibleTo<NoexceptFunction, char&>();
  CheckNotConvertibleTo<NoexceptFunction, char*>();

  // NoexceptFunction&
  CheckNotConvertibleTo<NoexceptFunction&, void>();
  CheckNotConvertibleTo<NoexceptFunction&, Function>();
  CheckNotConvertibleTo<NoexceptFunction&, NoexceptFunction>();
  CheckConvertibleTo<NoexceptFunction&, Function&>();
  CheckConvertibleTo<NoexceptFunction&, NoexceptFunction&>();

  CheckConvertibleTo<NoexceptFunction&, Function*>();
  CheckConvertibleTo<NoexceptFunction&, NoexceptFunction*>();
  CheckNotConvertibleTo<NoexceptFunction&, Array>();
  CheckNotConvertibleTo<NoexceptFunction&, Array&>();
  CheckNotConvertibleTo<NoexceptFunction&, char>();
  CheckNotConvertibleTo<NoexceptFunction&, char&>();
  CheckNotConvertibleTo<NoexceptFunction&, char*>();

  // NoexceptFunction*
  CheckNotConvertibleTo<NoexceptFunction*, void>();
  CheckNotConvertibleTo<NoexceptFunction*, Function>();
  CheckNotConvertibleTo<NoexceptFunction*, Function&>();
  CheckNotConvertibleTo<NoexceptFunction*, NoexceptFunction>();
  CheckNotConvertibleTo<NoexceptFunction*, NoexceptFunction&>();
  CheckConvertibleTo<NoexceptFunction*, Function*>();
  CheckConvertibleTo<NoexceptFunction*, NoexceptFunction*>();

  CheckNotConvertibleTo<NoexceptFunction*, Array>();
  CheckNotConvertibleTo<NoexceptFunction*, Array&>();
  CheckNotConvertibleTo<NoexceptFunction*, char>();
  CheckNotConvertibleTo<NoexceptFunction*, char&>();
  CheckNotConvertibleTo<NoexceptFunction*, char*>();

  // Array
  CheckNotConvertibleTo<Array, void>();
  CheckNotConvertibleTo<Array, Function>();
  CheckNotConvertibleTo<Array, Function&>();
  CheckNotConvertibleTo<Array, Function*>();
  CheckNotConvertibleTo<Array, NoexceptFunction>();
  CheckNotConvertibleTo<Array, NoexceptFunction&>();
  CheckNotConvertibleTo<Array, NoexceptFunction*>();
  CheckNotConvertibleTo<Array, Array>();

  static_assert(!std::convertible_to<Array, Array&>);
  static_assert(std::convertible_to<Array, const Array&>);
  static_assert(!std::convertible_to<Array, const volatile Array&>);

  static_assert(!std::convertible_to<const Array, Array&>);
  static_assert(std::convertible_to<const Array, const Array&>);
  static_assert(!std::convertible_to<Array, volatile Array&>);
  static_assert(!std::convertible_to<Array, const volatile Array&>);

  static_assert(std::convertible_to<Array, Array&&>);
  static_assert(std::convertible_to<Array, const Array&&>);
  static_assert(std::convertible_to<Array, volatile Array&&>);
  static_assert(std::convertible_to<Array, const volatile Array&&>);
  static_assert(std::convertible_to<const Array, const Array&&>);
  static_assert(!std::convertible_to<Array&, Array&&>);
  static_assert(!std::convertible_to<Array&&, Array&>);

  CheckNotConvertibleTo<Array, char>();
  CheckNotConvertibleTo<Array, char&>();

  static_assert(std::convertible_to<Array, char*>);
  static_assert(std::convertible_to<Array, const char*>);
  static_assert(std::convertible_to<Array, char* const>);
  static_assert(std::convertible_to<Array, char* const volatile>);

  static_assert(!std::convertible_to<const Array, char*>);
  static_assert(std::convertible_to<const Array, const char*>);

  static_assert(!std::convertible_to<char[42][42], char*>);
  static_assert(!std::convertible_to<char[][1], char*>);

  // Array&
  CheckNotConvertibleTo<Array&, void>();
  CheckNotConvertibleTo<Array&, Function>();
  CheckNotConvertibleTo<Array&, Function&>();
  CheckNotConvertibleTo<Array&, Function*>();
  CheckNotConvertibleTo<Array&, NoexceptFunction>();
  CheckNotConvertibleTo<Array&, NoexceptFunction&>();
  CheckNotConvertibleTo<Array&, NoexceptFunction*>();
  CheckNotConvertibleTo<Array&, Array>();

  static_assert(std::convertible_to<Array&, Array&>);
  static_assert(std::convertible_to<Array&, const Array&>);
  static_assert(!std::convertible_to<const Array&, Array&>);
  static_assert(std::convertible_to<const Array&, const Array&>);

  CheckNotConvertibleTo<Array&, char>();
  CheckNotConvertibleTo<Array&, char&>();

  static_assert(std::convertible_to<Array&, char*>);
  static_assert(std::convertible_to<Array&, const char*>);
  static_assert(!std::convertible_to<const Array&, char*>);
  static_assert(std::convertible_to<const Array&, const char*>);

  static_assert(std::convertible_to<Array, StringType>);
  static_assert(std::convertible_to<char(&)[], StringType>);

  // char
  CheckNotConvertibleTo<char, void>();
  CheckNotConvertibleTo<char, Function>();
  CheckNotConvertibleTo<char, Function&>();
  CheckNotConvertibleTo<char, Function*>();
  CheckNotConvertibleTo<char, NoexceptFunction>();
  CheckNotConvertibleTo<char, NoexceptFunction&>();
  CheckNotConvertibleTo<char, NoexceptFunction*>();
  CheckNotConvertibleTo<char, Array>();
  CheckNotConvertibleTo<char, Array&>();

  CheckConvertibleTo<char, char>();

  static_assert(!std::convertible_to<char, char&>);
  static_assert(std::convertible_to<char, const char&>);
  static_assert(!std::convertible_to<const char, char&>);
  static_assert(std::convertible_to<const char, const char&>);

  CheckNotConvertibleTo<char, char*>();

  // char&
  CheckNotConvertibleTo<char&, void>();
  CheckNotConvertibleTo<char&, Function>();
  CheckNotConvertibleTo<char&, Function&>();
  CheckNotConvertibleTo<char&, Function*>();
  CheckNotConvertibleTo<char&, NoexceptFunction>();
  CheckNotConvertibleTo<char&, NoexceptFunction&>();
  CheckNotConvertibleTo<char&, NoexceptFunction*>();
  CheckNotConvertibleTo<char&, Array>();
  CheckNotConvertibleTo<char&, Array&>();

  CheckConvertibleTo<char&, char>();

  static_assert(std::convertible_to<char&, char&>);
  static_assert(std::convertible_to<char&, const char&>);
  static_assert(!std::convertible_to<const char&, char&>);
  static_assert(std::convertible_to<const char&, const char&>);

  CheckNotConvertibleTo<char&, char*>();

  // char*
  CheckNotConvertibleTo<char*, void>();
  CheckNotConvertibleTo<char*, Function>();
  CheckNotConvertibleTo<char*, Function&>();
  CheckNotConvertibleTo<char*, Function*>();
  CheckNotConvertibleTo<char*, NoexceptFunction>();
  CheckNotConvertibleTo<char*, NoexceptFunction&>();
  CheckNotConvertibleTo<char*, NoexceptFunction*>();
  CheckNotConvertibleTo<char*, Array>();
  CheckNotConvertibleTo<char*, Array&>();

  CheckNotConvertibleTo<char*, char>();
  CheckNotConvertibleTo<char*, char&>();

  static_assert(std::convertible_to<char*, char*>);
  static_assert(std::convertible_to<char*, const char*>);
  static_assert(!std::convertible_to<const char*, char*>);
  static_assert(std::convertible_to<const char*, const char*>);

  // NonCopyable
  static_assert(std::convertible_to<NonCopyable&, NonCopyable&>);
  static_assert(std::convertible_to<NonCopyable&, const NonCopyable&>);
  static_assert(std::convertible_to<NonCopyable&, const volatile NonCopyable&>);
  static_assert(std::convertible_to<NonCopyable&, volatile NonCopyable&>);
  static_assert(std::convertible_to<const NonCopyable&, const NonCopyable&>);
  static_assert(
      std::convertible_to<const NonCopyable&, const volatile NonCopyable&>);
  static_assert(
      std::convertible_to<volatile NonCopyable&, const volatile NonCopyable&>);
  static_assert(std::convertible_to<const volatile NonCopyable&,
                                    const volatile NonCopyable&>);
  static_assert(!std::convertible_to<const NonCopyable&, NonCopyable&>);

  // This test requires Access control SFINAE which we only have in C++11 or when
  // we are using the compiler builtin for convertible_to.
  CheckNotConvertibleTo<NonCopyable&, NonCopyable>();

  // Ensure that CannotInstantiate is not instantiated by convertible_to when it is not needed.
  // For example CannotInstantiate is instantiated as a part of ADL lookup for arguments of type CannotInstantiate*.
  static_assert(
      std::convertible_to<CannotInstantiate<int>*, CannotInstantiate<int>*>);

  // Test for PR13592
  static_assert(!std::convertible_to<abstract, abstract>);

  CommonlyNotConvertibleTo<int>();
  CommonlyNotConvertibleTo<bool>();
  CommonlyNotConvertibleTo<nullptr_t>();

  CheckNotConvertibleTo<int, ExplicitlyConstructible>();
  CheckNotConvertibleTo<ExplicitlyConvertible, ExplicitlyConstructible>();
  CheckNotConvertibleTo<ExplicitlyConstructible, ExplicitlyConvertible>();
  CheckIsConvertibleButNotConvertibleTo<ImplicitlyConvertible,
                                        ExplicitlyConstructible>();
  CheckNotConvertibleTo<ImplicitlyConstructible, ImplicitlyConvertible>();

  return 0;
}
