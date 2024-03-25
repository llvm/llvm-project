//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <deque>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T>
class small_pointer {
  std::uint16_t offset;
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

  T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t);

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

  T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t);

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
