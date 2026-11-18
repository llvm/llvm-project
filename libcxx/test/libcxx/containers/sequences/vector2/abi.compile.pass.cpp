//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

#include <cstdint>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T>
class small_pointer {
public:
  using value_type        = T;
  using difference_type   = std::int16_t;
  using pointer           = small_pointer;
  using reference         = T&;
  using iterator_category = std::random_access_iterator_tag;

private:
  [[maybe_unused]] std::uint16_t offset;
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
  std::vector2<int, common_base_allocator<int> > v;
  [[no_unique_address]] common_base_allocator<int> a;
};

#if __SIZE_WIDTH__ == 64
// PROTOTYPE: std::vector2 uses the size-based layout, which stores the allocator as a plain [[no_unique_address]]
// member instead of using _LIBCPP_COMPRESSED_PAIR. This means that it suffers from https://llvm.org/PR154146, which
// was fixed for the pointer-based layout (std::vector has sizeof(user_struct) == 24). This is a pre-existing issue
// with _LIBCPP_ABI_VECTOR_LAYOUT_SIZE_BASED, unrelated to P3866R0.
static_assert(sizeof(user_struct) == 32, "");
static_assert(TEST_ALIGNOF(user_struct) == 8, "");

static_assert(sizeof(std::vector2<int>) == 24, "");
static_assert(sizeof(std::vector2<int, min_allocator<int> >) == 24, "");
static_assert(sizeof(std::vector2<int, test_allocator<int> >) == 32, ""); // std::vector: 40
static_assert(sizeof(std::vector2<int, small_iter_allocator<int> >) == 6, "");

static_assert(sizeof(std::vector2<char>) == 24, "");
static_assert(sizeof(std::vector2<char, min_allocator<char> >) == 24, "");
static_assert(sizeof(std::vector2<char, test_allocator<char> >) == 32, ""); // std::vector: 40
static_assert(sizeof(std::vector2<char, small_iter_allocator<char> >) == 6, "");

static_assert(TEST_ALIGNOF(std::vector2<int>) == 8, "");
static_assert(TEST_ALIGNOF(std::vector2<int, min_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(std::vector2<int, test_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(std::vector2<int, small_iter_allocator<int> >) == 2, "");

static_assert(TEST_ALIGNOF(std::vector2<char>) == 8, "");
static_assert(TEST_ALIGNOF(std::vector2<char, min_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(std::vector2<char, test_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(std::vector2<char, small_iter_allocator<char> >) == 2, "");

#elif __SIZE_WIDTH__ == 32
// PROTOTYPE: These values were copied from std::vector and have not been verified for std::vector2.
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 16, "");
#  else
static_assert(sizeof(user_struct) == 12, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 4, "");

static_assert(sizeof(std::vector2<int>) == 12, "");
static_assert(sizeof(std::vector2<int, min_allocator<int> >) == 12, "");
static_assert(sizeof(std::vector2<int, test_allocator<int> >) == 24, "");
static_assert(sizeof(std::vector2<int, small_iter_allocator<int> >) == 6, "");

static_assert(sizeof(std::vector2<char>) == 12, "");
static_assert(sizeof(std::vector2<char, min_allocator<char> >) == 12, "");
static_assert(sizeof(std::vector2<char, test_allocator<char> >) == 24, "");
static_assert(sizeof(std::vector2<char, small_iter_allocator<char> >) == 6, "");

static_assert(TEST_ALIGNOF(std::vector2<int>) == 4, "");
static_assert(TEST_ALIGNOF(std::vector2<int, min_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(std::vector2<int, test_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(std::vector2<int, small_iter_allocator<int> >) == 2, "");

static_assert(TEST_ALIGNOF(std::vector2<char>) == 4, "");
static_assert(TEST_ALIGNOF(std::vector2<char, min_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(std::vector2<char, test_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(std::vector2<char, small_iter_allocator<char> >) == 2, "");

#else
#  error std::size_t has an unexpected size
#endif
