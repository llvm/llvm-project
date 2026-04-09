//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-abi-fix-unordered-container-size-type, libcpp-abi-no-compressed-pair-padding

// std::unique_ptr is used as an implementation detail of the unordered containers, so the layout of
// unordered containers changes when bounded unique_ptr is enabled.
// UNSUPPORTED: libcpp-has-abi-bounded-unique_ptr

#include <cstdint>
#include <unordered_set>

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

template <class T, class Alloc>
using unordered_set_alloc = std::unordered_set<T, std::hash<T>, std::equal_to<T>, Alloc>;

struct user_struct {
  unordered_set_alloc<int, common_base_allocator<int> > v;
  [[no_unique_address]] common_base_allocator<int> a;
};

#if __SIZE_WIDTH__ == 64
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 48, "");
#  else
static_assert(sizeof(user_struct) == 40, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 8, "");

static_assert(sizeof(unordered_set_alloc<int, std::allocator<int> >) == 40, "");
static_assert(sizeof(unordered_set_alloc<int, min_allocator<int> >) == 40, "");
static_assert(sizeof(unordered_set_alloc<int, test_allocator<int> >) == 64, "");
static_assert(sizeof(unordered_set_alloc<int, small_iter_allocator<int> >) == 12, "");
static_assert(sizeof(unordered_set_alloc<int, final_small_iter_allocator<int> >) == 16, "");

static_assert(sizeof(unordered_set_alloc<char, std::allocator<char> >) == 40, "");
static_assert(sizeof(unordered_set_alloc<char, min_allocator<char> >) == 40, "");
static_assert(sizeof(unordered_set_alloc<char, test_allocator<char> >) == 64, "");
static_assert(sizeof(unordered_set_alloc<char, small_iter_allocator<char> >) == 12, "");
static_assert(sizeof(unordered_set_alloc<char, final_small_iter_allocator<char> >) == 16, "");

static_assert(TEST_ALIGNOF(unordered_set_alloc<int, std::allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, min_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, test_allocator<int> >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, small_iter_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, final_small_iter_allocator<int> >) == 4, "");

static_assert(TEST_ALIGNOF(unordered_set_alloc<char, std::allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, min_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, test_allocator<char> >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, small_iter_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, final_small_iter_allocator<char> >) == 4, "");

struct TEST_ALIGNAS(32) AlignedHash {};
struct UnalignedEqualTo {};

// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(std::unordered_set<int, AlignedHash, UnalignedEqualTo>) == 64, "");
#  else
static_assert(sizeof(std::unordered_set<int, AlignedHash, UnalignedEqualTo>) == 96, "");
#  endif
static_assert(TEST_ALIGNOF(std::unordered_set<int, AlignedHash, UnalignedEqualTo>) == 32, "");

#elif __SIZE_WIDTH__ == 32
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 24, "");
#  else
static_assert(sizeof(user_struct) == 20, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 4, "");

static_assert(sizeof(unordered_set_alloc<int, std::allocator<int> >) == 20, "");
static_assert(sizeof(unordered_set_alloc<int, min_allocator<int> >) == 20, "");
static_assert(sizeof(unordered_set_alloc<int, test_allocator<int> >) == 44, "");
static_assert(sizeof(unordered_set_alloc<int, small_iter_allocator<int> >) == 12, "");
static_assert(sizeof(unordered_set_alloc<int, final_small_iter_allocator<int> >) == 16, "");

static_assert(sizeof(unordered_set_alloc<char, std::allocator<char> >) == 20, "");
static_assert(sizeof(unordered_set_alloc<char, min_allocator<char> >) == 20, "");
static_assert(sizeof(unordered_set_alloc<char, test_allocator<char> >) == 44, "");
static_assert(sizeof(unordered_set_alloc<char, small_iter_allocator<char> >) == 12, "");
static_assert(sizeof(unordered_set_alloc<char, final_small_iter_allocator<char> >) == 16, "");

static_assert(TEST_ALIGNOF(unordered_set_alloc<int, std::allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, min_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, test_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, small_iter_allocator<int> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<int, final_small_iter_allocator<int> >) == 4, "");

static_assert(TEST_ALIGNOF(unordered_set_alloc<char, std::allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, min_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, test_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, small_iter_allocator<char> >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_set_alloc<char, final_small_iter_allocator<char> >) == 4, "");

struct TEST_ALIGNAS(32) AlignedHash {};
struct UnalignedEqualTo {};

// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(std::unordered_set<int, AlignedHash, UnalignedEqualTo>) == 64);
#  else
static_assert(sizeof(std::unordered_set<int, AlignedHash, UnalignedEqualTo>) == 96);
#  endif
static_assert(TEST_ALIGNOF(std::unordered_set<int, AlignedHash, UnalignedEqualTo>) == 32);

#else
#  error std::size_t has an unexpected size
#endif
