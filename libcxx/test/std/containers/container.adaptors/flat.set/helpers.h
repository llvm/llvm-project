//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_flat_set_HELPERS_H
#define SUPPORT_flat_set_HELPERS_H

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
#include <flat_set>

#include "test_allocator.h"
#include "test_macros.h"

template <class... Args>
void check_invariant(const std::flat_set<Args...>& m) {
  assert(std::is_sorted(m.begin(), m.end(), m.key_comp()));
  auto key_equal = [&](const auto& x, const auto& y) {
    const auto& c = m.key_comp();
    return !c(x, y) && !c(y, x);
  };
  assert(std::adjacent_find(m.begin(), m.end(), key_equal) == m.end());
}

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

  bool* transparent_used  = nullptr;
  TransparentComparator() = default;
  TransparentComparator(bool& used) : transparent_used(&used) {}

  template <class T, bool Convertible>
  bool operator()(const T& t, const Transparent<T, Convertible>& transparent) const {
    if (transparent_used != nullptr) {
      *transparent_used = true;
    }
    return t < transparent.t;
  }

  template <class T, bool Convertible>
  bool operator()(const Transparent<T, Convertible>& transparent, const T& t) const {
    if (transparent_used != nullptr) {
      *transparent_used = true;
    }
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

#endif

template <class F>
void test_emplace_exception_guarantee([[maybe_unused]] F&& emplace_function) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using C = TransparentComparator;
  {
    // Throw on emplace the key, and underlying has strong exception guarantee
    using KeyContainer = std::vector<int, test_allocator<int>>;
    using M            = std::flat_set<int, C, KeyContainer>;

    LIBCPP_STATIC_ASSERT(std::__container_traits<KeyContainer>::__emplacement_has_strong_exception_safety_guarantee);

    test_allocator_statistics stats;

    KeyContainer a({1, 2, 3, 4}, test_allocator<int>{&stats});
    [[maybe_unused]] auto expected_keys = a;
    M m(std::sorted_unique, std::move(a));

    stats.throw_after = 1;
    try {
      emplace_function(m, 0);
      assert(false);
    } catch (const std::bad_alloc&) {
      check_invariant(m);
      // In libc++, the flat_set is unchanged
      LIBCPP_ASSERT(m.size() == 4);
      LIBCPP_ASSERT(std::ranges::equal(m, expected_keys));
    }
  }
  {
    // Throw on emplace the key, and underlying has no strong exception guarantee
    using KeyContainer = EmplaceUnsafeContainer<int>;
    using M            = std::flat_set<int, C, KeyContainer>;

    LIBCPP_STATIC_ASSERT(!std::__container_traits<KeyContainer>::__emplacement_has_strong_exception_safety_guarantee);
    KeyContainer a = {1, 2, 3, 4};
    M m(std::sorted_unique, std::move(a));
    try {
      emplace_function(m, 0);
      assert(false);
    } catch (int) {
      check_invariant(m);
      // In libc++, the flat_set is cleared
      LIBCPP_ASSERT(m.size() == 0);
    }
  }
#endif
}

template <class F>
void test_insert_range_exception_guarantee([[maybe_unused]] F&& insert_function) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using KeyContainer = EmplaceUnsafeContainer<int>;
  using M            = std::flat_set<int, std::ranges::less, KeyContainer>;
  test_allocator_statistics stats;
  KeyContainer a{1, 2, 3, 4};
  M m(std::sorted_unique, std::move(a));

  std::vector<int> newValues = {0, 1, 5, 6, 7, 8};
  stats.throw_after          = 1;
  try {
    insert_function(m, newValues);
    assert(false);
  } catch (int) {
    check_invariant(m);
    // In libc++, we clear if anything goes wrong when inserting a range
    LIBCPP_ASSERT(m.size() == 0);
  }
#endif
}

template <class F>
void test_erase_exception_guarantee([[maybe_unused]] F&& erase_function) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    // key erase throws
    using KeyContainer = ThrowOnEraseContainer<int>;
    using M            = std::flat_set<int, TransparentComparator, KeyContainer>;

    KeyContainer a{1, 2, 3, 4};
    M m(std::sorted_unique, std::move(a));
    try {
      erase_function(m, 3);
      assert(false);
    } catch (int) {
      check_invariant(m);
      // In libc++, we clear if anything goes wrong when erasing
      LIBCPP_ASSERT(m.size() == 0);
    }
  }
#endif
}
class Moveable {
  int int_;
  double double_;

public:
  Moveable() : int_(0), double_(0) {}
  Moveable(int i, double d) : int_(i), double_(d) {}
  Moveable(Moveable&& x) : int_(x.int_), double_(x.double_) {
    x.int_    = -1;
    x.double_ = -1;
  }
  Moveable& operator=(Moveable&& x) {
    int_      = x.int_;
    x.int_    = -1;
    double_   = x.double_;
    x.double_ = -1;
    return *this;
  }

  Moveable(const Moveable&)            = delete;
  Moveable& operator=(const Moveable&) = delete;
  bool operator==(const Moveable& x) const { return int_ == x.int_ && double_ == x.double_; }
  bool operator<(const Moveable& x) const { return int_ < x.int_ || (int_ == x.int_ && double_ < x.double_); }

  int get() const { return int_; }
  bool moved() const { return int_ == -1; }
};

#endif // SUPPORT_flat_set_HELPERS_H
