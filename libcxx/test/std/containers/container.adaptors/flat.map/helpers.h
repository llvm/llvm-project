//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FLAT_MAP_HELPERS_H
#define SUPPORT_FLAT_MAP_HELPERS_H

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

#include "test_macros.h"

struct StartsWith {
  explicit StartsWith(char ch) : lower_(1, ch), upper_(1, ch + 1) {}
  StartsWith(const StartsWith&)     = delete;
  void operator=(const StartsWith&) = delete;
  struct Less {
    using is_transparent = void;
    bool operator()(const std::string& a, const std::string& b) const { return a < b; }
    bool operator()(const StartsWith& a, const std::string& b) const { return a.upper_ <= b; }
    bool operator()(const std::string& a, const StartsWith& b) const { return a < b.lower_; }
    bool operator()(const StartsWith&, const StartsWith&) const {
      assert(false); // should not be called
      return false;
    }
  };

private:
  std::string lower_;
  std::string upper_;
};

template <class T>
struct CopyOnlyVector : std::vector<T> {
  using std::vector<T>::vector;

  CopyOnlyVector(const CopyOnlyVector&) = default;
  CopyOnlyVector(CopyOnlyVector&& other) : CopyOnlyVector(other) {}
  CopyOnlyVector(CopyOnlyVector&& other, std::vector<T>::allocator_type alloc) : CopyOnlyVector(other, alloc) {}

  CopyOnlyVector& operator=(const CopyOnlyVector&) = default;
  CopyOnlyVector& operator=(CopyOnlyVector& other) { return this->operator=(other); }
};

template <class T, bool ConvertibleToT = false>
struct Transparent {
  T t;

  operator T() const
    requires ConvertibleToT
  {
    return t;
  }
};

template <class T>
using ConvertibleTransparent = Transparent<T, true>;

template <class T>
using NonConvertibleTransparent = Transparent<T, false>;

struct TransparentComparator {
  using is_transparent = void;
  template <class T, bool Convertible>
  bool operator()(const T& t, const Transparent<T, Convertible>& transparent) const {
    return t < transparent.t;
  }

  template <class T, bool Convertible>
  bool operator()(const Transparent<T, Convertible>& transparent, const T& t) const {
    return transparent.t < t;
  }

  template <class T>
  bool operator()(const T& t1, const T& t2) const {
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

template <bool Copyable = true>
struct ThrowOnMove {
  int i;
  ThrowOnMove(int ii) : i(ii) {}
  ThrowOnMove(const ThrowOnMove&)
    requires Copyable
  = default;

  ThrowOnMove& operator=(const ThrowOnMove&)
    requires Copyable
  = default;

  ThrowOnMove(ThrowOnMove&&) { throw 42; }

  ThrowOnMove& operator=(ThrowOnMove&& other) {
    other.i = -1;
    throw 42;
  }

  friend bool operator==(const ThrowOnMove&, const ThrowOnMove&)  = default;
  friend auto operator<=>(const ThrowOnMove&, const ThrowOnMove&) = default;
};

using CopyableThrowOnMove = ThrowOnMove<true>;
using MoveOnlyThrowOnMove = ThrowOnMove<false>;

struct ThrowOnSecondMove {
  int i;
  int count;
  ThrowOnSecondMove(int ii) : i(ii), count(0) {}
  ThrowOnSecondMove(const ThrowOnSecondMove&)            = delete;
  ThrowOnSecondMove& operator=(const ThrowOnSecondMove&) = delete;

  ThrowOnSecondMove(ThrowOnSecondMove&& other) : i(other.i), count(other.count) {
    if (++count > 1)
      throw 42;
  }

  ThrowOnSecondMove& operator=(ThrowOnSecondMove&&) {
    if (++count > 1)
      throw 42;
    return *this;
  }

  friend bool operator==(const ThrowOnSecondMove&, const ThrowOnSecondMove&)  = default;
  friend auto operator<=>(const ThrowOnSecondMove&, const ThrowOnSecondMove&) = default;
};

template <class T, class Compare = std::less<>>
bool is_sorted_and_unique(T&& container, Compare compare = Compare()) {
  auto greater_or_equal_to = [&](const auto& x, const auto& y) { return !compare(x, y); };
  return std::ranges::adjacent_find(container, greater_or_equal_to) == std::ranges::end(container);
}

#endif // SUPPORT_FLAT_MAP_HELPERS_H
