//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STABLE_SORT_HELPERS_H
#define SUPPORT_STABLE_SORT_HELPERS_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "test_macros.h"

#if TEST_STD_VER < 11
#  error "C++11 or greater is required to use this header"
#endif

namespace support {

// A linear congruential generator, so that every platform sorts the same sequences.
class KeyGenerator {
  unsigned long long state_;

public:
  TEST_CONSTEXPR explicit KeyGenerator(unsigned seed) : state_(seed) {}

  TEST_CONSTEXPR_CXX14 int below(int bound) {
    state_ = (state_ * 1664525ull + 1013904223ull) & 0xffffffffull;
    return static_cast<int>((state_ >> 8) % static_cast<unsigned long long>(bound));
  }
};

// Block merge sorts behave differently depending on how many distinct keys the runs they merge contain, so several
// patterns only differ in how many distinct keys they have and where those are.
enum KeyPattern {
  ascending_keys,
  descending_keys,
  equal_keys,
  random_keys,
  two_keys,
  three_keys,
  root_keys,
  double_root_keys,
  quarter_keys,
  sawtooth_keys,
  descending_sawtooth_keys,
  organ_pipe_keys,
  jittered_ascending_keys,
  jittered_descending_keys,
  random_tail_keys,
  descending_plateau_keys,
  descending_pair_keys,
  distinct_window_keys,
  equal_then_random_keys,
  random_then_equal_keys,
  key_pattern_count
};

TEST_CONSTEXPR_CXX20 inline std::vector<int> make_keys(KeyPattern pattern, int size, unsigned seed = 0) {
  KeyGenerator random(seed * 31u + static_cast<unsigned>(pattern));
  int root = 1;
  while ((root + 1) * (root + 1) <= size)
    ++root;

  std::vector<int> keys;
  keys.reserve(static_cast<std::size_t>(size));
  for (int i = 0; i != size; ++i) {
    switch (pattern) {
    case ascending_keys:
      keys.push_back(i);
      break;
    case descending_keys:
      keys.push_back(size - i);
      break;
    case equal_keys:
      keys.push_back(0);
      break;
    case random_keys:
      keys.push_back(random.below(1 << 30));
      break;
    case two_keys:
      keys.push_back(random.below(2));
      break;
    case three_keys:
      keys.push_back(random.below(3));
      break;
    case root_keys:
      keys.push_back(random.below(root));
      break;
    case double_root_keys:
      keys.push_back(random.below(2 * root + 1));
      break;
    case quarter_keys:
      keys.push_back(random.below(size / 4 + 1));
      break;
    case sawtooth_keys:
      keys.push_back(i % 7);
      break;
    case descending_sawtooth_keys:
      keys.push_back((size - i) % 11);
      break;
    case organ_pipe_keys:
      keys.push_back(i < size - i ? i : size - i);
      break;
    case jittered_ascending_keys:
      keys.push_back(i + random.below(5));
      break;
    case jittered_descending_keys:
      keys.push_back(size - i + random.below(5));
      break;
    case random_tail_keys:
      keys.push_back(i < size - size / 5 ? i : random.below(size));
      break;
    case descending_plateau_keys:
      keys.push_back((size - i) / 37);
      break;
    case descending_pair_keys:
      keys.push_back((size - i) / 2);
      break;
    case distinct_window_keys:
      keys.push_back(size / 3 <= i && i < size / 3 + size / 8 ? 2 + i : random.below(2));
      break;
    case equal_then_random_keys:
      keys.push_back(i < size / 2 ? 1 << 30 : random.below(1 << 30));
      break;
    case random_then_equal_keys:
      keys.push_back(i < size / 2 ? random.below(1 << 30) : -1);
      break;
    case key_pattern_count:
      break;
    }
  }
  return keys;
}

typedef std::pair<int, int> KeyAndIndex;

// The only arrangement that a stable sort by key can produce when the i-th element of the input has the key keys[i].
// It is computed with an unstable sort so that it doesn't depend on the algorithm under test.
TEST_CONSTEXPR_CXX20 inline std::vector<KeyAndIndex> stable_order(const std::vector<int>& keys) {
  std::vector<KeyAndIndex> order;
  order.reserve(keys.size());
  for (std::size_t i = 0; i != keys.size(); ++i)
    order.push_back(KeyAndIndex(keys[i], static_cast<int>(i)));
  std::sort(order.begin(), order.end());
  return order;
}

// A trivially copyable element that isn't default constructible. Implementations pick different strategies depending
// on the size of the elements, which the padding controls.
template <std::size_t Padding>
struct Element {
  int key;
  int index;
  unsigned char padding[Padding];

  TEST_CONSTEXPR Element(int k, int i) : key(k), index(i), padding() {}
};

template <>
struct Element<0> {
  int key;
  int index;

  TEST_CONSTEXPR Element(int k, int i) : key(k), index(i) {}
};

template <std::size_t Padding>
TEST_CONSTEXPR int key_of(const Element<Padding>& element) {
  return element.key;
}

template <std::size_t Padding>
TEST_CONSTEXPR int index_of(const Element<Padding>& element) {
  return element.index;
}

// Ignores the index, so that elements with the same key are equivalent without being equal.
template <std::size_t Padding>
TEST_CONSTEXPR bool operator<(const Element<Padding>& a, const Element<Padding>& b) {
  return a.key < b.key;
}

struct KeyLess {
  template <class T, class U>
  TEST_CONSTEXPR bool operator()(const T& a, const U& b) const {
    return key_of(a) < key_of(b);
  }
};

template <class T>
TEST_CONSTEXPR_CXX20 std::vector<T> make_elements(const std::vector<int>& keys) {
  std::vector<T> elements;
  elements.reserve(keys.size());
  for (std::size_t i = 0; i != keys.size(); ++i)
    elements.push_back(T(keys[i], static_cast<int>(i)));
  return elements;
}

template <class Range>
TEST_CONSTEXPR_CXX20 std::vector<KeyAndIndex> keys_and_indices(const Range& range) {
  std::vector<KeyAndIndex> result;
  for (auto it = range.begin(); it != range.end(); ++it)
    result.push_back(KeyAndIndex(key_of(*it), index_of(*it)));
  return result;
}

struct ProbeState {
  // The number of probes that have been constructed and not destroyed yet.
  static int alive;
  // The number of moves of probes that succeed before one throws. Moves never throw while this is negative.
  static int moves_before_throw;
};

int ProbeState::alive              = 0;
int ProbeState::moves_before_throw = -1;

struct ProbeException {};

// A move-only element that isn't default constructible. It asserts when an algorithm does something that the
// requirements on the elements of a sorted range don't allow it to rely on: using the value of an element that was
// moved from, moving an element into itself, or treating raw memory as an element.
template <std::size_t Padding>
class Probe {
  enum : unsigned { has_value = 0x600D, moved_from_state = 0x30FED, destroyed = 0xDEAD };

  int key_;
  int index_;
  // Not an enumeration, because the assertions also read it from raw memory, where an enumeration may not have a value
  // that can be read at all.
  unsigned state_;
  unsigned char padding_[Padding + 1];

  static void count_move() {
    if (ProbeState::moves_before_throw == 0) {
      ProbeState::moves_before_throw = -1;
      TEST_THROW(ProbeException());
    }
    if (ProbeState::moves_before_throw > 0)
      --ProbeState::moves_before_throw;
  }

public:
  Probe(int key, int index) : key_(key), index_(index), state_(has_value), padding_() { ++ProbeState::alive; }

  Probe(const Probe&)            = delete;
  Probe& operator=(const Probe&) = delete;

  Probe(Probe&& other) : key_(other.key_), index_(other.index_), state_(has_value), padding_() {
    assert(other.state_ == has_value);
    count_move();
    other.state_ = moved_from_state;
    ++ProbeState::alive;
  }

  Probe& operator=(Probe&& other) {
    assert(this != std::addressof(other));
    assert(state_ == has_value || state_ == moved_from_state);
    assert(other.state_ == has_value);
    count_move();
    key_         = other.key_;
    index_       = other.index_;
    state_       = has_value;
    other.state_ = moved_from_state;
    return *this;
  }

  ~Probe() {
    assert(state_ == has_value || state_ == moved_from_state);
    state_ = destroyed;
    --ProbeState::alive;
  }

  bool moved_from() const { return state_ == moved_from_state; }

  int key() const {
    assert(state_ == has_value);
    return key_;
  }

  int index() const {
    assert(state_ == has_value);
    return index_;
  }
};

template <std::size_t Padding>
int key_of(const Probe<Padding>& probe) {
  return probe.key();
}

template <std::size_t Padding>
int index_of(const Probe<Padding>& probe) {
  return probe.index();
}

struct CheckedIteratorState {
  // The number of operations that CheckedIterators may still perform, so that a test can notice a loop that doesn't
  // terminate.
  static long long operations_left;
};

long long CheckedIteratorState::operations_left = 1LL << 62;

// A random access iterator that asserts when it leaves its range, when it is dereferenced at the end of its range, and
// when it is combined with a singular iterator or with an iterator into another range. It only supports arithmetic with
// its difference type, which is all that an iterator has to support.
template <class T, class Difference = std::ptrdiff_t>
class CheckedIterator {
  T* current_;
  T* first_;
  T* last_;
  bool singular_;

  void count_operation() const {
    assert(!singular_);
    --CheckedIteratorState::operations_left;
    assert(CheckedIteratorState::operations_left >= 0);
  }

  void count_operation(const CheckedIterator& other) const {
    count_operation();
    assert(!other.singular_);
    assert(first_ == other.first_);
    assert(last_ == other.last_);
  }

public:
  typedef std::random_access_iterator_tag iterator_category;
  typedef T value_type;
  typedef Difference difference_type;
  typedef T* pointer;
  typedef T& reference;

  CheckedIterator() : current_(nullptr), first_(nullptr), last_(nullptr), singular_(true) {}
  CheckedIterator(T* current, T* first, T* last) : current_(current), first_(first), last_(last), singular_(false) {}

  reference operator*() const {
    count_operation();
    assert(current_ != last_);
    return *current_;
  }

  pointer operator->() const { return std::addressof(**this); }
  reference operator[](difference_type n) const { return *(*this + n); }

  CheckedIterator& operator+=(difference_type n) {
    count_operation();
    assert(first_ - current_ <= n);
    assert(n <= last_ - current_);
    current_ += n;
    return *this;
  }

  CheckedIterator& operator-=(difference_type n) {
    count_operation();
    assert(current_ - last_ <= n);
    assert(n <= current_ - first_);
    current_ -= n;
    return *this;
  }

  CheckedIterator& operator++() { return *this += difference_type(1); }
  CheckedIterator& operator--() { return *this -= difference_type(1); }

  CheckedIterator operator++(int) {
    CheckedIterator copy(*this);
    ++*this;
    return copy;
  }

  CheckedIterator operator--(int) {
    CheckedIterator copy(*this);
    --*this;
    return copy;
  }

  friend CheckedIterator operator+(CheckedIterator it, difference_type n) {
    it += n;
    return it;
  }

  friend CheckedIterator operator+(difference_type n, CheckedIterator it) {
    it += n;
    return it;
  }

  friend CheckedIterator operator-(CheckedIterator it, difference_type n) {
    it -= n;
    return it;
  }

  friend difference_type operator-(const CheckedIterator& a, const CheckedIterator& b) {
    a.count_operation(b);
    return static_cast<difference_type>(a.current_ - b.current_);
  }

  friend bool operator==(const CheckedIterator& a, const CheckedIterator& b) {
    a.count_operation(b);
    return a.current_ == b.current_;
  }

  friend bool operator!=(const CheckedIterator& a, const CheckedIterator& b) {
    a.count_operation(b);
    return a.current_ != b.current_;
  }

  friend bool operator<(const CheckedIterator& a, const CheckedIterator& b) {
    a.count_operation(b);
    return a.current_ < b.current_;
  }

  friend bool operator>(const CheckedIterator& a, const CheckedIterator& b) {
    a.count_operation(b);
    return a.current_ > b.current_;
  }

  friend bool operator<=(const CheckedIterator& a, const CheckedIterator& b) {
    a.count_operation(b);
    return a.current_ <= b.current_;
  }

  friend bool operator>=(const CheckedIterator& a, const CheckedIterator& b) {
    a.count_operation(b);
    return a.current_ >= b.current_;
  }

  template <class Other>
  void operator+=(Other) = delete;
  template <class Other>
  void operator-=(Other) = delete;
  template <class Other>
  friend void operator+(CheckedIterator, Other) = delete;
  template <class Other>
  friend void operator+(Other, CheckedIterator) = delete;
  template <class Other>
  friend void operator-(CheckedIterator, Other) = delete;
};

} // namespace support

#endif // SUPPORT_STABLE_SORT_HELPERS_H
