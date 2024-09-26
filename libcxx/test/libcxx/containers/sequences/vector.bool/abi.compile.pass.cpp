//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-abi-no-compressed-pair-padding

#include <vector>
#include <cstdint>

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
  using size_type       = std::int16_t;
  using difference_type = std::int16_t;

  small_iter_allocator() TEST_NOEXCEPT {}

  template <class U>
  small_iter_allocator(small_iter_allocator<U>) TEST_NOEXCEPT {}

  T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t);

  friend bool operator==(small_iter_allocator, small_iter_allocator) { return true; }
  friend bool operator!=(small_iter_allocator, small_iter_allocator) { return false; }
};

#if __SIZE_WIDTH__ == 64

static_assert(sizeof(std::vector<bool>) == 24, "");
static_assert(sizeof(std::vector<bool, min_allocator<bool> >) == 24, "");
static_assert(sizeof(std::vector<bool, test_allocator<bool> >) == 40, "");
static_assert(sizeof(std::vector<bool, small_iter_allocator<bool> >) == 6, "");

static_assert(TEST_ALIGNOF(std::vector<bool>) == 8, "");
static_assert(TEST_ALIGNOF(std::vector<bool, min_allocator<bool> >) == 8, "");
static_assert(TEST_ALIGNOF(std::vector<bool, test_allocator<bool> >) == 8, "");
static_assert(TEST_ALIGNOF(std::vector<bool, small_iter_allocator<bool> >) == 2, "");

#elif __SIZE_WIDTH__ == 32

static_assert(sizeof(std::vector<bool>) == 12, "");
static_assert(sizeof(std::vector<bool, min_allocator<bool> >) == 12, "");
static_assert(sizeof(std::vector<bool, test_allocator<bool> >) == 24, "");
static_assert(sizeof(std::vector<bool, small_iter_allocator<bool> >) == 6, "");

static_assert(TEST_ALIGNOF(std::vector<bool>) == 4, "");
static_assert(TEST_ALIGNOF(std::vector<bool, min_allocator<bool> >) == 4, "");
static_assert(TEST_ALIGNOF(std::vector<bool, test_allocator<bool> >) == 4, "");
static_assert(TEST_ALIGNOF(std::vector<bool, small_iter_allocator<bool> >) == 2, "");

#else
#  error std::size_t has an unexpected size
#endif
