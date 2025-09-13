//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_CONTAINER_ADAPTORS_FLAT_HELPERS_H
#define TEST_STD_CONTAINERS_CONTAINER_ADAPTORS_FLAT_HELPERS_H

#include <cstdint>
#include <vector>

#include "test_macros.h"

template <class T>
struct CopyOnlyVector : std::vector<T> {
  using std::vector<T>::vector;

  constexpr CopyOnlyVector(const CopyOnlyVector&) = default;
  constexpr CopyOnlyVector(CopyOnlyVector&& other) : CopyOnlyVector(other) {}
  constexpr CopyOnlyVector(CopyOnlyVector&& other, std::vector<T>::allocator_type alloc)
      : CopyOnlyVector(other, alloc) {}

  constexpr CopyOnlyVector& operator=(const CopyOnlyVector&) = default;
  constexpr CopyOnlyVector& operator=(CopyOnlyVector& other) { return this->operator=(other); }
};

template <class T>
struct SillyReserveVector : std::vector<T> {
  using std::vector<T>::vector;

  constexpr void reserve(std::size_t) { this->clear(); }
};

template <class T, bool ConvertibleToT = false>
struct Transparent {
  T t;

  constexpr explicit operator T() const
    requires ConvertibleToT
  {
    return t;
  }
};

template <class T>
using ConvertibleTransparent = Transparent<T, true>;

template <class T>
using ExplicitlyConvertibleTransparent = Transparent<T, true>;

template <class T>
using NonConvertibleTransparent = Transparent<T, false>;

struct TransparentComparator {
  using is_transparent = void;

  bool* transparent_used  = nullptr;
  TransparentComparator() = default;
  constexpr TransparentComparator(bool& used) : transparent_used(&used) {}

  template <class T, bool Convertible>
  constexpr bool operator()(const T& t, const Transparent<T, Convertible>& transparent) const {
    if (transparent_used != nullptr) {
      *transparent_used = true;
    }
    return t < transparent.t;
  }

  template <class T, bool Convertible>
  constexpr bool operator()(const Transparent<T, Convertible>& transparent, const T& t) const {
    if (transparent_used != nullptr) {
      *transparent_used = true;
    }
    return transparent.t < t;
  }

  template <class T>
  constexpr bool operator()(const T& t1, const T& t2) const {
    return t1 < t2;
  }
};

struct NonTransparentComparator {
  template <class T, bool Convertible>
  bool operator()(const T&, const Transparent<T, Convertible>&) const;

  template <class T, bool Convertible>
  bool operator()(const Transparent<T, Convertible>&, const T&) const;

  template <class T>
  bool operator()(const T&, const T&) const;
};

struct NoDefaultCtr {
  NoDefaultCtr() = delete;
};

class Moveable {
  int int_;
  double double_;

public:
  TEST_CONSTEXPR Moveable() : int_(0), double_(0) {}
  TEST_CONSTEXPR Moveable(int i, double d) : int_(i), double_(d) {}
  TEST_CONSTEXPR Moveable(Moveable&& x) : int_(x.int_), double_(x.double_) {
    x.int_    = -1;
    x.double_ = -1;
  }
  TEST_CONSTEXPR Moveable& operator=(Moveable&& x) {
    int_      = x.int_;
    x.int_    = -1;
    double_   = x.double_;
    x.double_ = -1;
    return *this;
  }

  Moveable(const Moveable&)            = delete;
  Moveable& operator=(const Moveable&) = delete;
  TEST_CONSTEXPR bool operator==(const Moveable& x) const { return int_ == x.int_ && double_ == x.double_; }
  TEST_CONSTEXPR bool operator<(const Moveable& x) const {
    return int_ < x.int_ || (int_ == x.int_ && double_ < x.double_);
  }

  TEST_CONSTEXPR int get() const { return int_; }
  TEST_CONSTEXPR bool moved() const { return int_ == -1; }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
template <class T>
struct EmplaceUnsafeContainer : std::vector<T> {
  using std::vector<T>::vector;

  template <class... Args>
  auto emplace(Args&&... args) -> decltype(std::declval<std::vector<T>>().emplace(std::forward<Args>(args)...)) {
    if (this->size() > 1) {
      auto it1 = this->begin();
      auto it2 = it1 + 1;
      // messing up the container
      std::iter_swap(it1, it2);
    }

    throw 42;
  }

  template <class... Args>
  auto insert(Args&&... args) -> decltype(std::declval<std::vector<T>>().insert(std::forward<Args>(args)...)) {
    if (this->size() > 1) {
      auto it1 = this->begin();
      auto it2 = it1 + 1;
      // messing up the container
      std::iter_swap(it1, it2);
    }

    throw 42;
  }

  template <class... Args>
  auto insert_range(Args&&... args)
      -> decltype(std::declval<std::vector<T>>().insert_range(std::forward<Args>(args)...)) {
    if (this->size() > 1) {
      auto it1 = this->begin();
      auto it2 = it1 + 1;
      // messing up the container
      std::iter_swap(it1, it2);
    }

    throw 42;
  }
};

template <class T>
struct ThrowOnEraseContainer : std::vector<T> {
  using std::vector<T>::vector;

  template <class... Args>
  auto erase(Args&&... args) -> decltype(std::declval<std::vector<T>>().erase(std::forward<Args>(args)...)) {
    throw 42;
  }
};

template <class T>
struct ThrowOnMoveContainer : std::vector<T> {
  using std::vector<T>::vector;

  ThrowOnMoveContainer(ThrowOnMoveContainer&&) { throw 42; }

  ThrowOnMoveContainer& operator=(ThrowOnMoveContainer&&) { throw 42; }
};

#endif // TEST_HAS_NO_EXCEPTIONS

#endif // TEST_STD_CONTAINERS_CONTAINER_ADAPTORS_FLAT_HELPERS_H