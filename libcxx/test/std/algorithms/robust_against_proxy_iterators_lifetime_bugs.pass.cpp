//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Making this test file support C++03 is difficult; the lack of support for initializer lists is a major issue.
// UNSUPPORTED: c++03

// <algorithm>

#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <set>

#include "test_macros.h"

// This file contains checks for lifetime issues across all the classic algorithms. It uses two complementary
// approaches:
// - runtime checks using a proxy iterator that tracks the lifetime of itself and its objects to catch potential
//   lifetime issues;
// - `constexpr` checks using a `constexpr`-friendly proxy iterator that catch undefined behavior.

// A random-access proxy iterator that tracks the lifetime of itself and its `value_type` and `reference` objects to
// prevent potential lifetime issues in algorithms.
//
// This class cannot be `constexpr` because its cache is a static variable. The cache cannot be provided as
// a constructor parameter because `LifetimeIterator` has to be default-constructible.
class LifetimeIterator {
  // The cache simply tracks addresses of the local variables.
  class LifetimeCache {
    std::set<const void*> cache_;

  public:
    bool contains(const void* ptr) const { return cache_.find(ptr) != cache_.end(); }

    void insert(const void* ptr) {
      assert(!contains(ptr));
      cache_.insert(ptr);
    }

    void erase(const void* ptr) {
      assert(contains(ptr));
      cache_.erase(ptr);
    }
  };

 public:
  struct Value {
    int i_;
    bool moved_from_ = false; // Check for double moves and reads after moving.

    Value() { lifetime_cache.insert(this); }
    Value(int i) : i_(i) { lifetime_cache.insert(this); }
    ~Value() { lifetime_cache.erase(this); }

    Value(const Value& rhs) : i_(rhs.i_) {
      assert(lifetime_cache.contains(&rhs));
      assert(!rhs.moved_from_);

      lifetime_cache.insert(this);
    }

    Value(Value&& rhs) noexcept : i_(rhs.i_) {
      assert(lifetime_cache.contains(&rhs));

      assert(!rhs.moved_from_);
      rhs.moved_from_ = true;

      // It's ok if it throws -- since it's a test, terminating the program is acceptable.
      lifetime_cache.insert(this);
    }

    Value& operator=(const Value& rhs) {
      assert(lifetime_cache.contains(this) && lifetime_cache.contains(&rhs));
      assert(!rhs.moved_from_);

      i_ = rhs.i_;
      moved_from_ = false;

      return *this;
    }

    Value& operator=(Value&& rhs) noexcept {
      assert(lifetime_cache.contains(this) && lifetime_cache.contains(&rhs));

      assert(!rhs.moved_from_);
      rhs.moved_from_ = true;

      i_ = rhs.i_;
      moved_from_ = false;

      return *this;
    }

    friend bool operator<(const Value& lhs, const Value& rhs) {
      assert(lifetime_cache.contains(&lhs) && lifetime_cache.contains(&rhs));
      assert(!lhs.moved_from_ && !rhs.moved_from_);

      return lhs.i_ < rhs.i_;
    }

    friend bool operator==(const Value& lhs, const Value& rhs) {
      assert(lifetime_cache.contains(&lhs) && lifetime_cache.contains(&rhs));
      assert(!lhs.moved_from_ && !rhs.moved_from_);

      return lhs.i_ == rhs.i_;
    }

  };

  struct Reference {
    Value* v_;
    bool moved_from_ = false; // Check for double moves and reads after moving.

    Reference(Value& v) : v_(&v) {
      lifetime_cache.insert(this);
    }

    ~Reference() {
      lifetime_cache.erase(this);
    }

    Reference(const Reference& rhs) : v_(rhs.v_) {
      assert(lifetime_cache.contains(&rhs));
      assert(!rhs.moved_from_);

      lifetime_cache.insert(this);
    }

    Reference(Reference&& rhs) noexcept : v_(rhs.v_) {
      assert(lifetime_cache.contains(&rhs));

      assert(!rhs.moved_from_);
      rhs.moved_from_ = true;

      lifetime_cache.insert(this);
    }

    Reference& operator=(const Reference& rhs) {
      assert(lifetime_cache.contains(this) && lifetime_cache.contains(&rhs));
      assert(!rhs.moved_from_);

      v_ = rhs.v_;
      moved_from_ = false;

      return *this;
    }

    Reference& operator=(Reference&& rhs) noexcept {
      assert(lifetime_cache.contains(this) && lifetime_cache.contains(&rhs));

      assert(!rhs.moved_from_);
      rhs.moved_from_ = true;

      v_ = rhs.v_;
      moved_from_ = false;

      return *this;
    }

    operator Value() const {
      assert(lifetime_cache.contains(this));
      assert(!moved_from_);

      return *v_;
    }

    Reference& operator=(Value v) {
      assert(lifetime_cache.contains(this));
      assert(!moved_from_);

      *v_ = v;
      moved_from_ = false;

      return *this;
    }

    friend bool operator<(const Reference& lhs, const Reference& rhs) {
      assert(lifetime_cache.contains(&lhs) && lifetime_cache.contains(&rhs));
      assert(!lhs.moved_from_ && !rhs.moved_from_);

      return *lhs.v_ < *rhs.v_;
    }

    friend bool operator==(const Reference& lhs, const Reference& rhs) {
      assert(lifetime_cache.contains(&lhs) && lifetime_cache.contains(&rhs));
      assert(!lhs.moved_from_ && !rhs.moved_from_);

      return *lhs.v_ == *rhs.v_;
    }

    friend void swap(Reference lhs, Reference rhs) {
      assert(lifetime_cache.contains(&lhs) && lifetime_cache.contains(&rhs));
      assert(!lhs.moved_from_ && !rhs.moved_from_);

      std::swap(*(lhs.v_), *(rhs.v_));
    }
  };

  using difference_type   = int;
  using value_type        = Value;
  using reference         = Reference;
  using pointer           = void;
  using iterator_category = std::random_access_iterator_tag;

  Value* ptr_ = nullptr;
  bool moved_from_ = false; // Check for double moves and reads after moving.

  LifetimeIterator() = default;
  LifetimeIterator(Value* ptr) : ptr_(ptr) {}

  LifetimeIterator(const LifetimeIterator& rhs) : ptr_(rhs.ptr_) {
    assert(!rhs.moved_from_);
  }

  LifetimeIterator& operator=(const LifetimeIterator& rhs) {
    assert(!rhs.moved_from_);

    ptr_ = rhs.ptr_;
    moved_from_ = false;

    return *this;
  }

  LifetimeIterator(LifetimeIterator&& rhs) noexcept : ptr_(rhs.ptr_) {
    assert(!rhs.moved_from_);
    rhs.moved_from_ = true;
    rhs.ptr_ = nullptr;
  }

  LifetimeIterator& operator=(LifetimeIterator&& rhs) noexcept {
    assert(!rhs.moved_from_);
    rhs.moved_from_ = true;
    moved_from_ = false;

    ptr_ = rhs.ptr_;
    rhs.ptr_ = nullptr;

    return *this;
  }

  Reference operator*() const {
    assert(!moved_from_);
    return Reference(*ptr_);
  }

  LifetimeIterator& operator++() {
    assert(!moved_from_);

    ++ptr_;
    return *this;
  }

  LifetimeIterator operator++(int) {
    assert(!moved_from_);

    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend bool operator==(const LifetimeIterator& lhs, const LifetimeIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ == rhs.ptr_;
  }
  friend bool operator!=(const LifetimeIterator& lhs, const LifetimeIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ != rhs.ptr_;
  }

  LifetimeIterator& operator--() {
    assert(!moved_from_);

    --ptr_;
    return *this;
  }

  LifetimeIterator operator--(int) {
    assert(!moved_from_);

    auto tmp = *this;
    --*this;
    return tmp;
  }

  LifetimeIterator& operator+=(difference_type n) {
    assert(!moved_from_);

    ptr_ += n;
    return *this;
  }

  LifetimeIterator& operator-=(difference_type n) {
    assert(!moved_from_);

    ptr_ -= n;
    return *this;
  }

  Reference operator[](difference_type i) const {
    assert(!moved_from_);
    return Reference(*(ptr_ + i));
  }

  friend bool operator<(const LifetimeIterator& lhs, const LifetimeIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ < rhs.ptr_;
  }

  friend bool operator>(const LifetimeIterator& lhs, const LifetimeIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ > rhs.ptr_;
  }

  friend bool operator<=(const LifetimeIterator& lhs, const LifetimeIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ <= rhs.ptr_;
  }

  friend bool operator>=(const LifetimeIterator& lhs, const LifetimeIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ >= rhs.ptr_;
  }

  friend LifetimeIterator operator+(const LifetimeIterator& lhs, difference_type n) {
    assert(!lhs.moved_from_);
    return LifetimeIterator(lhs.ptr_ + n);
  }

  friend LifetimeIterator operator+(difference_type n, const LifetimeIterator& lhs) {
    assert(!lhs.moved_from_);
    return LifetimeIterator(n + lhs.ptr_);
  }

  friend LifetimeIterator operator-(const LifetimeIterator& lhs, difference_type n) {
    assert(!lhs.moved_from_);
    return LifetimeIterator(lhs.ptr_ - n);
  }

  friend difference_type operator-(LifetimeIterator lhs, LifetimeIterator rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return static_cast<int>(lhs.ptr_ - rhs.ptr_);
  }

  static LifetimeCache lifetime_cache;
};

LifetimeIterator::LifetimeCache LifetimeIterator::lifetime_cache;

#if TEST_STD_VER > 17
// A constexpr-friendly proxy iterator to check for undefined behavior in algorithms (since undefined behavior is
// statically caught in `constexpr` context).
class ConstexprIterator {
 public:
  struct Reference {
    int* v_;
    bool moved_from_ = false; // Check for double moves and reads after moving.

    constexpr Reference(int& v) : v_(&v) { }

    constexpr Reference(const Reference& rhs) = default;
    constexpr Reference& operator=(const Reference& rhs) {
      assert(!rhs.moved_from_);
      v_ = rhs.v_;
      moved_from_ = false;

      return *this;
    }

    constexpr Reference(Reference&& rhs) noexcept : v_(rhs.v_) {
      assert(!rhs.moved_from_);
      rhs.moved_from_ = true;
    }

    constexpr Reference& operator=(Reference&& rhs) noexcept {
      assert(!rhs.moved_from_);
      rhs.moved_from_ = true;
      moved_from_ = false;

      v_ = rhs.v_;
      return *this;
    }

    constexpr operator int() const {
      assert(!moved_from_);
      return *v_;
    }

    constexpr Reference& operator=(int v) {
      *v_ = v;
      moved_from_ = false;

      return *this;
    }

    friend constexpr bool operator<(const Reference& lhs, const Reference& rhs) {
      assert(!lhs.moved_from_ && !rhs.moved_from_);
      return *lhs.v_ < *rhs.v_;
    }

    friend constexpr bool operator==(const Reference& lhs, const Reference& rhs) {
      assert(!lhs.moved_from_ && !rhs.moved_from_);
      return *lhs.v_ == *rhs.v_;
    }

    friend constexpr void swap(Reference lhs, Reference rhs) {
      assert(!lhs.moved_from_ && !rhs.moved_from_);
      std::swap(*(lhs.v_), *(rhs.v_));
    }
  };

  using difference_type   = int;
  using value_type        = int;
  using reference         = Reference;
  using pointer           = void;
  using iterator_category = std::random_access_iterator_tag;

  int* ptr_ = nullptr;
  bool moved_from_ = false; // Check for double moves and reads after moving.

  constexpr ConstexprIterator() = default;
  constexpr ConstexprIterator(int* ptr) : ptr_(ptr) {}

  constexpr ConstexprIterator(const ConstexprIterator& rhs) : ptr_(rhs.ptr_) {
    assert(!rhs.moved_from_);
  }

  constexpr ConstexprIterator& operator=(const ConstexprIterator& rhs) {
    assert(!rhs.moved_from_);

    ptr_ = rhs.ptr_;
    moved_from_ = false;

    return *this;
  }

  constexpr ConstexprIterator(ConstexprIterator&& rhs) noexcept : ptr_(rhs.ptr_) {
    assert(!rhs.moved_from_);
    rhs.moved_from_ = true;
    rhs.ptr_ = nullptr;
  }

  constexpr ConstexprIterator& operator=(ConstexprIterator&& rhs) noexcept {
    assert(!rhs.moved_from_);
    rhs.moved_from_ = true;
    moved_from_ = false;

    ptr_ = rhs.ptr_;
    rhs.ptr_ = nullptr;

    return *this;
  }

  constexpr Reference operator*() const {
    assert(!moved_from_);
    return Reference(*ptr_);
  }

  constexpr ConstexprIterator& operator++() {
    assert(!moved_from_);

    ++ptr_;
    return *this;
  }

  constexpr ConstexprIterator operator++(int) {
    assert(!moved_from_);

    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend constexpr bool operator==(const ConstexprIterator& lhs, const ConstexprIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ == rhs.ptr_;
  }

  friend constexpr bool operator!=(const ConstexprIterator& lhs, const ConstexprIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ != rhs.ptr_;
  }

  constexpr ConstexprIterator& operator--() {
    assert(!moved_from_);

    --ptr_;
    return *this;
  }

  constexpr ConstexprIterator operator--(int) {
    assert(!moved_from_);

    auto tmp = *this;
    --*this;
    return tmp;
  }

  constexpr ConstexprIterator& operator+=(difference_type n) {
    assert(!moved_from_);

    ptr_ += n;
    return *this;
  }

  constexpr ConstexprIterator& operator-=(difference_type n) {
    assert(!moved_from_);

    ptr_ -= n;
    return *this;
  }

  constexpr Reference operator[](difference_type i) const {
    return Reference(*(ptr_ + i));
  }

  friend constexpr auto operator<=>(const ConstexprIterator& lhs, const ConstexprIterator& rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return lhs.ptr_ <=> rhs.ptr_;
  }

  friend constexpr ConstexprIterator operator+(const ConstexprIterator& lhs, difference_type n) {
    assert(!lhs.moved_from_);
    return ConstexprIterator(lhs.ptr_ + n);
  }

  friend constexpr ConstexprIterator operator+(difference_type n, const ConstexprIterator& lhs) {
    assert(!lhs.moved_from_);
    return ConstexprIterator(n + lhs.ptr_);
  }

  friend constexpr ConstexprIterator operator-(const ConstexprIterator& lhs, difference_type n) {
    assert(!lhs.moved_from_);
    return ConstexprIterator(lhs.ptr_ - n);
  }

  friend constexpr difference_type operator-(ConstexprIterator lhs, ConstexprIterator rhs) {
    assert(!lhs.moved_from_ && !rhs.moved_from_);
    return static_cast<int>(lhs.ptr_ - rhs.ptr_);
  }
};

#endif // TEST_STD_VER > 17

template <class T, size_t N = 32>
class Input {
  using Array = std::array<T, N>;

  size_t size_ = 0;
  Array values_ = {};

public:
  template <size_t N2>
  TEST_CONSTEXPR_CXX20 Input(std::array<T, N2> from) {
    static_assert(N2 <= N, "");

    std::copy(from.begin(), from.end(), begin());
    size_ = N2;
  }

  TEST_CONSTEXPR_CXX20 typename Array::iterator begin() { return values_.begin(); }
  TEST_CONSTEXPR_CXX20 typename Array::iterator end() { return values_.begin() + size_; }
  TEST_CONSTEXPR_CXX20 size_t size() const { return size_; }
};

// TODO: extend `Value` and `Reference` so that it's possible to pass plain integers to all the algorithms.

// Several generic inputs that are useful for many algorithms. Provides two unsorted sequences with and without
// duplicates, with positive and negative values; and a few corner cases, like an empty sequence, a sequence of all
// duplicates, and so on.
template <class Iter>
TEST_CONSTEXPR_CXX20 std::array<Input<typename Iter::value_type>, 8> get_simple_in() {
  using T = typename Iter::value_type;
  std::array<Input<T>, 8> result = {
    Input<T>({std::array<T, 0>{ }}),
    Input<T>({std::array<T, 1>{ T{1} }}),
    Input<T>({std::array<T, 1>{ T{-1} }}),
    Input<T>({std::array<T, 2>{ T{-1}, {1} }}),
    Input<T>({std::array<T, 3>{ T{1}, {1}, {1} }}),
    Input<T>({std::array<T, 3>{ T{-1}, {-1}, {-1} }}),
    Input<T>({std::array<T, 9>{ T{-8}, {6}, {3}, {2}, {1}, {5}, {-4}, {-9}, {3} }}),
    Input<T>({std::array<T, 9>{ T{-8}, {3}, {3}, {2}, {5}, {-4}, {-4}, {-4}, {1} }}),
  };
  return result;
}

// Sorted inputs of varying lengths.
template <class Iter>
TEST_CONSTEXPR_CXX20 std::array<Input<typename Iter::value_type>, 8> get_sorted_in() {
  using T = typename Iter::value_type;
  std::array<Input<T>, 8> result = {
    Input<T>({std::array<T, 0>{ }}),
    Input<T>({std::array<T, 1>{ T{1} }}),
    Input<T>({std::array<T, 1>{ T{-1} }}),
    Input<T>({std::array<T, 2>{ T{-1}, {1} }}),
    Input<T>({std::array<T, 3>{ T{1}, {1}, {1} }}),
    Input<T>({std::array<T, 3>{ T{-1}, {-1}, {-1} }}),
    Input<T>({std::array<T, 8>{ T{-8}, {-5}, {-3}, {-1}, {1}, {4}, {5}, {9} }}),
    Input<T>({std::array<T, 11>{ T{-8}, {-5}, {-3}, {-3}, {-1}, {1}, {4}, {5}, {5}, {9}, {9} }}),
  };
  return result;
}

// Inputs for testing `std::sort`. These have been manually verified to exercise all internal functions in `std::sort`
// except the branchless sort ones (which can't be triggered with proxy arrays).
template <class Iter>
TEST_CONSTEXPR_CXX20 std::array<Input<typename Iter::value_type>, 8> get_sort_test_in() {
  using T = typename Iter::value_type;
  std::array<Input<T>, 8> result = {
    Input<T>({std::array<T, 0>{ }}),
    Input<T>({std::array<T, 1>{ T{1} }}),
    Input<T>({std::array<T, 1>{ T{-1} }}),
    Input<T>({std::array<T, 2>{ T{-1}, {1} }}),
    Input<T>({std::array<T, 3>{ T{1}, {1}, {1} }}),
    Input<T>({std::array<T, 3>{ T{-1}, {-1}, {-1} }}),
    Input<T>({std::array<T, 8>{ T{-8}, {-5}, {-3}, {-1}, {1}, {4}, {5}, {9} }}),
    Input<T>({std::array<T, 11>{ T{-8}, {-5}, {-3}, {-3}, {-1}, {1}, {4}, {5}, {5}, {9}, {9} }}),
  };
  return result;
}

template <class Input, size_t N, class Func>
TEST_CONSTEXPR_CXX20 void test(std::array<Input, N> inputs, Func func) {
  for (auto&& in : inputs) {
    func(in.begin(), in.end());
  }
}

template <class Input, size_t N, class Func>
TEST_CONSTEXPR_CXX20 void test_n(std::array<Input, N> inputs, Func func) {
  for (auto&& in : inputs) {
    func(in.begin(), in.size());
  }
}

constexpr int to_int(int x) { return x; }
int to_int(LifetimeIterator::Value x) { return x.i_; }

std::mt19937 rand_gen() { return std::mt19937(); }

template <class Iter>
TEST_CONSTEXPR_CXX20 bool test() {
  using T = typename Iter::value_type;

  auto is_neg = [](const T& val) { return to_int(val) < 0; };
  auto gen = [] { return T{42}; };
  auto identity = [] (T val) -> T { return val; };

  constexpr int N = 32;
  std::array<T, N> output;
  auto out = output.begin();
  T x{1};
  T y{3};

  auto simple_in = get_simple_in<Iter>();
  auto sorted_in = get_sorted_in<Iter>();
  auto sort_test_in = get_sort_test_in<Iter>();

  using I = Iter;

  test(simple_in, [&](I b, I e) { std::any_of(b, e, is_neg); });
  test(simple_in, [&](I b, I e) { std::all_of(b, e, is_neg); });
  test(simple_in, [&](I b, I e) { std::none_of(b, e, is_neg); });
  test(simple_in, [&](I b, I e) { std::find(b, e, T{1}); });
  test(simple_in, [&](I b, I e) { std::find_if(b, e, is_neg); });
  test(simple_in, [&](I b, I e) { std::find_if_not(b, e, is_neg); });
  // TODO: find_first_of
  test(simple_in, [&](I b, I e) { std::adjacent_find(b, e); });
  // TODO: mismatch
  // TODO: equal
  // TODO: lexicographical_compare
  // TODO: partition_point
  test(sorted_in, [&](I b, I e) { std::lower_bound(b, e, x); });
  test(sorted_in, [&](I b, I e) { std::upper_bound(b, e, x); });
  test(sorted_in, [&](I b, I e) { std::equal_range(b, e, x); });
  test(sorted_in, [&](I b, I e) { std::binary_search(b, e, x); });
  // `min`, `max` and `minmax` don't use iterators.
  test(simple_in, [&](I b, I e) { std::min_element(b, e); });
  test(simple_in, [&](I b, I e) { std::max_element(b, e); });
  test(simple_in, [&](I b, I e) { std::minmax_element(b, e); });
  test(simple_in, [&](I b, I e) { std::count(b, e, x); });
  test(simple_in, [&](I b, I e) { std::count_if(b, e, is_neg); });
  // TODO: search
  // TODO: search_n
  // TODO: find_end
  // TODO: is_partitioned
  // TODO: is_sorted
  // TODO: is_sorted_until
  // TODO: includes
  // TODO: is_heap
  // TODO: is_heap_until
  // `clamp` doesn't use iterators.
  // TODO: is_permutation
  test(simple_in, [&](I b, I e) { std::for_each(b, e, is_neg); });
#if TEST_STD_VER > 14
  test_n(simple_in, [&](I b, size_t n) { std::for_each_n(b, n, is_neg); });
#endif
  test(simple_in, [&](I b, I e) { std::copy(b, e, out); });
  test_n(simple_in, [&](I b, size_t n) { std::copy_n(b, n, out); });
  test(simple_in, [&](I b, I e) { std::copy_backward(b, e, out + N); });
  test(simple_in, [&](I b, I e) { std::copy_if(b, e, out, is_neg); });
  test(simple_in, [&](I b, I e) { std::move(b, e, out); });
  test(simple_in, [&](I b, I e) { std::move_backward(b, e, out + N); });
  test(simple_in, [&](I b, I e) { std::transform(b, e, out, identity); });
  test(simple_in, [&](I b, I e) { std::generate(b, e, gen); });
  test_n(simple_in, [&](I b, size_t n) { std::generate_n(b, n, gen); });
  test(simple_in, [&](I b, I e) { std::remove_copy(b, e, out, x); });
  test(simple_in, [&](I b, I e) { std::remove_copy_if(b, e, out, is_neg); });
  test(simple_in, [&](I b, I e) { std::replace(b, e, x, y); });
  test(simple_in, [&](I b, I e) { std::replace_if(b, e, is_neg, y); });
  test(simple_in, [&](I b, I e) { std::replace_copy(b, e, out, x, y); });
  test(simple_in, [&](I b, I e) { std::replace_copy_if(b, e, out, is_neg, y); });
  // TODO: swap_ranges
  test(simple_in, [&](I b, I e) { std::reverse_copy(b, e, out); });
  // TODO: rotate_copy
  // TODO: sample
  // TODO: unique_copy
  // TODO: partition_copy
  // TODO: partial_sort_copy
  // TODO: merge
  // TODO: set_difference
  // TODO: set_intersection
  // TODO: set_symmetric_difference
  // TODO: set_union
  test(simple_in, [&](I b, I e) { std::remove(b, e, x); });
  test(simple_in, [&](I b, I e) { std::remove_if(b, e, is_neg); });
  test(simple_in, [&](I b, I e) { std::reverse(b, e); });
  // TODO: rotate
  if (!TEST_IS_CONSTANT_EVALUATED)
    test(simple_in, [&](I b, I e) { std::shuffle(b, e, rand_gen()); });
  // TODO: unique
  test(simple_in, [&](I b, I e) { std::partition(b, e, is_neg); });
  if (!TEST_IS_CONSTANT_EVALUATED)
    test(simple_in, [&](I b, I e) { std::stable_partition(b, e, is_neg); });
  if (!TEST_IS_CONSTANT_EVALUATED)
    test(sort_test_in, [&](I b, I e) { std::sort(b, e); });
  if (!TEST_IS_CONSTANT_EVALUATED)
    test(sort_test_in, [&](I b, I e) { std::stable_sort(b, e); });
  // TODO: partial_sort
  // TODO: nth_element
  // TODO: inplace_merge
  test(simple_in, [&](I b, I e) { std::make_heap(b, e); });
  // TODO: push_heap
  // TODO: pop_heap
  // TODO: sort_heap
  test(simple_in, [&](I b, I e) { std::prev_permutation(b, e); });
  test(simple_in, [&](I b, I e) { std::next_permutation(b, e); });

  // TODO: algorithms in `<numeric>`
  // TODO: algorithms in `<memory>`

  return true;
}

void test_all() {
  test<LifetimeIterator>();
#if TEST_STD_VER > 17 // Most algorithms are only `constexpr` starting from C++20.
  static_assert(test<ConstexprIterator>());
#endif
}

int main(int, char**) {
  test_all();

  return 0;
}
