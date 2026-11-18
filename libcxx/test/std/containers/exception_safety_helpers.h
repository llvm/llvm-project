//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_EXCEPTION_SAFETY_HELPERS_H
#define SUPPORT_EXCEPTION_SAFETY_HELPERS_H

#include <cassert>
#include <cstddef>
#include <forward_list>
#include <functional>
#include <utility>
#include "test_macros.h"

#if !defined(TEST_HAS_NO_EXCEPTIONS)
template <int NDefault, int NInplace, int NCopy, int NMove>
struct ThrowingBase {
  static bool throwing_enabled;
  static int default_constructed;
  static int created_inplace;
  static int created_by_copying;
  static int created_by_moving;
  static int destroyed;
  int x = 0; // Allows distinguishing between different instances.

  ThrowingBase() {
    ++default_constructed;
    if (throwing_enabled && default_constructed == NDefault) {
      throw -1;
    }
  }
  ThrowingBase(int value) : x(value) {
    ++created_inplace;
    if (throwing_enabled && created_inplace == NInplace) {
      throw -1;
    }
  }
  ~ThrowingBase() { ++destroyed; }

  ThrowingBase(const ThrowingBase& other) : x(other.x) {
    ++created_by_copying;
    if (throwing_enabled && created_by_copying == NCopy) {
      throw -1;
    }
  }

  ThrowingBase(ThrowingBase&& other) : x(other.x) {
    ++created_by_moving;
    if (throwing_enabled && created_by_moving == NMove) {
      throw -1;
    }
  }

  // Defined to silence GCC warnings. For test purposes, only copy construction is considered `created_by_copying`.
  ThrowingBase& operator=(const ThrowingBase& other) {
    x = other.x;
    return *this;
  }
  ThrowingBase& operator=(ThrowingBase&& other) {
    x = other.x;
    return *this;
  }

  friend bool operator==(const ThrowingBase& lhs, const ThrowingBase& rhs) { return lhs.x == rhs.x; }
  friend bool operator<(const ThrowingBase& lhs, const ThrowingBase& rhs) { return lhs.x < rhs.x; }

  static void reset() {
    default_constructed = created_inplace = created_by_copying = created_by_moving = destroyed = 0;
  }
};

template <int NDefault, int NInplace, int NCopy, int NMove>
bool ThrowingBase<NDefault, NInplace, NCopy, NMove>::throwing_enabled = true;
template <int NDefault, int NInplace, int NCopy, int NMove>
int ThrowingBase<NDefault, NInplace, NCopy, NMove>::default_constructed = 0;
template <int NDefault, int NInplace, int NCopy, int NMove>
int ThrowingBase<NDefault, NInplace, NCopy, NMove>::created_inplace = 0;
template <int NDefault, int NInplace, int NCopy, int NMove>
int ThrowingBase<NDefault, NInplace, NCopy, NMove>::created_by_copying = 0;
template <int NDefault, int NInplace, int NCopy, int NMove>
int ThrowingBase<NDefault, NInplace, NCopy, NMove>::created_by_moving = 0;
template <int NDefault, int NInplace, int NCopy, int NMove>
int ThrowingBase<NDefault, NInplace, NCopy, NMove>::destroyed = 0;

template <int NDefault, int NInplace, int NCopy, int NMove>
struct std::hash<ThrowingBase<NDefault, NInplace, NCopy, NMove>> {
  std::size_t operator()(const ThrowingBase<NDefault, NInplace, NCopy, NMove>& value) const { return value.x; }
};

template <int N>
using ThrowingDefault = ThrowingBase<N, 0, 0, 0>;

template <int N>
using ThrowingInplace = ThrowingBase<0, N, 0, 0>;

template <int N>
using ThrowingCopy = ThrowingBase<0, 0, N, 0>;

template <int N>
using ThrowingMove = ThrowingBase<0, 0, 0, N>;

template <int ThrowOn, int Size, class Func>
void test_exception_safety_throwing_default(Func&& func) {
  using T = ThrowingDefault<ThrowOn>;
  T::reset();

  try {
    func(Size);
    assert(false); // The function call above should throw.

  } catch (int) {
    assert(T::default_constructed == ThrowOn);
    assert(T::destroyed == ThrowOn - 1); // No destructor call for the partially-constructed element.
  }
}

template <int ThrowOn, int Size, class Func>
void test_exception_safety_throwing_copy(Func&& func) {
  using T = ThrowingCopy<ThrowOn>;
  T::reset();
  T in[Size];

  try {
    func(in, in + Size);
    assert(false); // The function call above should throw.

  } catch (int) {
    assert(T::created_by_copying == ThrowOn);
    assert(T::destroyed == ThrowOn - 1); // No destructor call for the partially-constructed element.
  }
}

// Destroys the container outside the user callback to avoid destroying extra elements upon throwing (which would
// complicate asserting that the expected number of elements was destroyed).
template <class Container, int ThrowOn, int Size, class Func>
void test_exception_safety_throwing_copy_container(Func&& func) {
  using T             = ThrowingCopy<ThrowOn>;
  T::throwing_enabled = false;
  T in[Size];
  Container c(in, in + Size);
  T::throwing_enabled = true;
  T::reset();

  try {
    func(std::move(c));
    assert(false); // The function call above should throw.

  } catch (int) {
    assert(T::created_by_copying == ThrowOn);
    assert(T::destroyed == ThrowOn - 1); // No destructor call for the partially-constructed element.
  }
}

template <int ThrowOnDefault, int ThrowOnInplace, int ThrowOnCopy, int ThrowOnMove, int Size, class Func>
void test_strong_exception_safety(Func&& func) {
  using T             = ThrowingBase<ThrowOnDefault, ThrowOnInplace, ThrowOnCopy, ThrowOnMove>;
  T::throwing_enabled = false;

  std::forward_list<T> c0(Size);
  for (int i = 0; i < Size; ++i)
    c0.emplace_front(i);
  std::forward_list<T> c = c0;
  T in[Size];
  T::reset();
  T::throwing_enabled = true;
  try {
    func(c, in, in + Size);
    assert(false); // The function call above should throw.

  } catch (int) {
    assert(T::default_constructed == ThrowOnDefault);
    assert(T::created_inplace == ThrowOnInplace);
    assert(T::created_by_copying == ThrowOnCopy);
    assert(T::created_by_moving == ThrowOnMove);
    assert(T::destroyed == ThrowOnDefault + ThrowOnInplace + ThrowOnCopy + ThrowOnMove -
                               1); // No destructor call for the partially-constructed element.
    assert(c == c0);               // Strong exception guarantee
  }
}

template <int ThrowOn, int Size, class Func>
void test_strong_exception_safety_throwing_inplace(Func&& func) {
  test_strong_exception_safety<0, ThrowOn, 0, 0, Size, Func>(std::forward<Func>(func));
}

template <int ThrowOn, int Size, class Func>
void test_strong_exception_safety_throwing_copy(Func&& func) {
  test_strong_exception_safety<0, 0, ThrowOn, 0, Size, Func>(std::forward<Func>(func));
}

template <int ThrowOn, int Size, class Func>
void test_strong_exception_safety_throwing_move(Func&& func) {
  test_strong_exception_safety<0, 0, 0, ThrowOn, Size, Func>(std::forward<Func>(func));
}

#endif // !defined(TEST_HAS_NO_EXCEPTIONS)

#endif // SUPPORT_EXCEPTION_SAFETY_HELPERS_H
