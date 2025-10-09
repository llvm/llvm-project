//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-abi-no-compressed-pair-padding

#include <cstdint>
#include <list>

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

struct allocator_base {};

// Make sure that types with a common base type don't get broken. See https://llvm.org/PR154146
template <class T>
struct common_base_allocator : allocator_base {
  using value_type = T;

  common_base_allocator() TEST_NOEXCEPT {}

  template <class U>
  common_base_allocator(common_base_allocator<U>) TEST_NOEXCEPT {}

  T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t);

  friend bool operator==(common_base_allocator, common_base_allocator) { return true; }
  friend bool operator!=(common_base_allocator, common_base_allocator) { return false; }
};

struct user_struct {
  std::list<int, common_base_allocator<int> > v;
  [[no_unique_address]] common_base_allocator<int> a;
};

#if __SIZE_WIDTH__ == 64
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 32, "");
#  else
static_assert(sizeof(user_struct) == 24, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 8, "");

static_assert(sizeof(std::list<int>) == 24, "");
static_assert(sizeof(std::list<int, min_allocator<int> >) == 24, "");
static_assert(sizeof(std::list<int, test_allocator<int> >) == 40, "");
static_assert(sizeof(std::list<int, small_iter_allocator<int> >) == 6, "");

static_assert(sizeof(std::list<char>) == 24, "");
static_assert(sizeof(std::list<char, min_allocator<char> >) == 24, "");
static_assert(sizeof(std::list<char, test_allocator<char> >) == 40, "");
static_assert(sizeof(std::list<char, small_iter_allocator<char> >) == 6, "");

static_assert(TEST_ALIGNOF(std::list<int>) == 8, "");
static_assert(TEST_ALIGNOF(std::list<int, min_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(std::list<int, test_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(std::list<int, small_iter_allocator<int> >) == 2, "");

static_assert(TEST_ALIGNOF(std::list<char>) == 8, "");
static_assert(TEST_ALIGNOF(std::list<char, min_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(std::list<char, test_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(std::list<char, small_iter_allocator<char> >) == 2, "");

#elif __SIZE_WIDTH__ == 32
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 16, "");
#  else
static_assert(sizeof(user_struct) == 12, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 4, "");

static_assert(sizeof(std::list<int>) == 12, "");
static_assert(sizeof(std::list<int, min_allocator<int> >) == 12, "");
static_assert(sizeof(std::list<int, test_allocator<int> >) == 24, "");
static_assert(sizeof(std::list<int, small_iter_allocator<int> >) == 6, "");

static_assert(sizeof(std::list<char>) == 12, "");
static_assert(sizeof(std::list<char, min_allocator<char> >) == 12, "");
static_assert(sizeof(std::list<char, test_allocator<char> >) == 24, "");
static_assert(sizeof(std::list<char, small_iter_allocator<char> >) == 6, "");

static_assert(TEST_ALIGNOF(std::list<int>) == 4, "");
static_assert(TEST_ALIGNOF(std::list<int, min_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(std::list<int, test_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(std::list<int, small_iter_allocator<int> >) == 2, "");

static_assert(TEST_ALIGNOF(std::list<char>) == 4, "");
static_assert(TEST_ALIGNOF(std::list<char, min_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(std::list<char, test_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(std::list<char, small_iter_allocator<char> >) == 2, "");

#else
#  error std::size_t has an unexpected size
#endif
