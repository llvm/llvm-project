//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-abi-no-compressed-pair-padding

#include <cstdint>
#include <deque>
#include <iterator>
#include <type_traits>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T>
class small_pointer {
  std::uint16_t offset;

public:
  using value_type        = typename std::remove_cv<T>::type;
  using difference_type   = std::int16_t;
  using reference         = T&;
  using pointer           = T*;
  using iterator_category = std::random_access_iterator_tag;

  template <class CT,
            typename std::enable_if<std::is_same<const T, CT>::value && !std::is_const<T>::value, int>::type = 0>
  operator small_pointer<CT>() const;
  template <
      class Void,
      typename std::enable_if<std::is_same<Void, void>::value && std::is_convertible<T*, void*>::value, int>::type = 0>
  operator small_pointer<Void>() const;
  template <
      class CVoid,
      typename std::enable_if<std::is_same<CVoid, const void>::value && std::is_convertible<T*, const void*>::value,
                              int>::type = 0>
  operator small_pointer<CVoid>() const;

  T& operator*() const;
  T* operator->() const;
  T& operator[](difference_type) const;

  small_pointer& operator++();
  small_pointer operator++(int);
  small_pointer& operator--();
  small_pointer operator--(int);
  small_pointer& operator+=(difference_type);
  small_pointer& operator-=(difference_type);

  friend small_pointer operator+(small_pointer, difference_type) { return small_pointer{}; }
  friend small_pointer operator+(difference_type, small_pointer) { return small_pointer{}; }
  friend small_pointer operator-(small_pointer, difference_type) { return small_pointer{}; }
  friend difference_type operator-(small_pointer, small_pointer) { return 0; }

  friend bool operator==(small_pointer, small_pointer) { return true; }
#if TEST_STD_VER < 20
  friend bool operator!=(small_pointer, small_pointer) { return false; }
#endif
  friend bool operator<(small_pointer, small_pointer) { return false; }
  friend bool operator>=(small_pointer, small_pointer) { return true; }
  friend bool operator>(small_pointer, small_pointer) { return false; }
  friend bool operator>=(small_pointer, small_pointer) { return true; }

  small_pointer pointer_to(T&);
};

template <>
class small_pointer<const void> {
  std::uint16_t offset;

public:
  template <class CT, typename std::enable_if<std::is_convertible<CT*, const void*>::value, int>::type = 0>
  explicit operator small_pointer<CT>() const;
};

template <>
class small_pointer<void*> {
  std::uint16_t offset;

public:
  operator small_pointer<const void>() const;

  template <class T, typename std::enable_if<std::is_convertible<T*, void*>::value, int>::type = 0>
  explicit operator small_pointer<T>() const;
};

template <class T>
class small_iter_allocator {
public:
  using value_type      = T;
  using pointer         = small_pointer<T>;
  using size_type       = std::uint16_t;
  using difference_type = std::int16_t;

  small_iter_allocator() TEST_NOEXCEPT {}

  template <class U>
  small_iter_allocator(small_iter_allocator<U>) TEST_NOEXCEPT {}

  pointer allocate(std::size_t n);
  void deallocate(pointer p, std::size_t);

  friend bool operator==(small_iter_allocator, small_iter_allocator) { return true; }
  friend bool operator!=(small_iter_allocator, small_iter_allocator) { return false; }
};

template <class T>
class final_small_iter_allocator final {
public:
  using value_type      = T;
  using pointer         = small_pointer<T>;
  using size_type       = std::uint16_t;
  using difference_type = std::int16_t;

  final_small_iter_allocator() TEST_NOEXCEPT {}

  template <class U>
  final_small_iter_allocator(final_small_iter_allocator<U>) TEST_NOEXCEPT {}

  pointer allocate(std::size_t n);
  void deallocate(pointer p, std::size_t);

  friend bool operator==(final_small_iter_allocator, final_small_iter_allocator) { return true; }
  friend bool operator!=(final_small_iter_allocator, final_small_iter_allocator) { return false; }
};

#if __SIZE_WIDTH__ == 64

static_assert(sizeof(std::deque<int>) == 48, "");
static_assert(sizeof(std::deque<int, min_allocator<int> >) == 48, "");
static_assert(sizeof(std::deque<int, test_allocator<int> >) == 80, "");
static_assert(sizeof(std::deque<int, small_iter_allocator<int> >) == 12, "");
static_assert(sizeof(std::deque<int, final_small_iter_allocator<int> >) == 16, "");

static_assert(sizeof(std::deque<char>) == 48, "");
static_assert(sizeof(std::deque<char, min_allocator<char> >) == 48, "");
static_assert(sizeof(std::deque<char, test_allocator<char> >) == 80, "");
static_assert(sizeof(std::deque<char, small_iter_allocator<char> >) == 12, "");
static_assert(sizeof(std::deque<char, final_small_iter_allocator<char> >) == 16, "");

static_assert(TEST_ALIGNOF(std::deque<int>) == 8, "");
static_assert(TEST_ALIGNOF(std::deque<int, min_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(std::deque<int, test_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(std::deque<int, small_iter_allocator<int> >) == 2, "");
static_assert(TEST_ALIGNOF(std::deque<int, final_small_iter_allocator<int> >) == 2, "");

static_assert(TEST_ALIGNOF(std::deque<char>) == 8, "");
static_assert(TEST_ALIGNOF(std::deque<char, min_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(std::deque<char, test_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(std::deque<char, small_iter_allocator<char> >) == 2, "");
static_assert(TEST_ALIGNOF(std::deque<char, final_small_iter_allocator<char> >) == 2, "");

#elif __SIZE_WIDTH__ == 32

static_assert(sizeof(std::deque<int>) == 24, "");
static_assert(sizeof(std::deque<int, min_allocator<int> >) == 24, "");
static_assert(sizeof(std::deque<int, test_allocator<int> >) == 48, "");
static_assert(sizeof(std::deque<int, small_iter_allocator<int> >) == 12, "");
static_assert(sizeof(std::deque<int, final_small_iter_allocator<int> >) == 16, "");

static_assert(sizeof(std::deque<char>) == 24, "");
static_assert(sizeof(std::deque<char, min_allocator<char> >) == 24, "");
static_assert(sizeof(std::deque<char, test_allocator<char> >) == 48, "");
static_assert(sizeof(std::deque<char, small_iter_allocator<char> >) == 12, "");
static_assert(sizeof(std::deque<char, final_small_iter_allocator<char> >) == 16, "");

static_assert(TEST_ALIGNOF(std::deque<int>) == 4, "");
static_assert(TEST_ALIGNOF(std::deque<int, min_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(std::deque<int, test_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(std::deque<int, small_iter_allocator<int> >) == 2, "");
static_assert(TEST_ALIGNOF(std::deque<int, final_small_iter_allocator<int> >) == 2, "");

static_assert(TEST_ALIGNOF(std::deque<char>) == 4, "");
static_assert(TEST_ALIGNOF(std::deque<char, min_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(std::deque<char, test_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(std::deque<char, small_iter_allocator<char> >) == 2, "");
static_assert(TEST_ALIGNOF(std::deque<char, final_small_iter_allocator<char> >) == 2, "");

#else
#  error std::size_t has an unexpected size
#endif
