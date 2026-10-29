//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-abi-no-compressed-pair-padding

#include <cstdint>
#include <map>

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
using map_alloc = std::map<T, T, std::less<T>, Alloc>;

struct user_struct {
  map_alloc<int, common_base_allocator<std::pair<const int, int> > > v;
  [[no_unique_address]] common_base_allocator<int> a;
};

struct TEST_ALIGNAS(32) AlignedLess {};
struct FinalLess final {};
struct NonEmptyLess {
  int i;
  char c;
};

static_assert(std::is_empty<std::__map_value_compare<int, std::pair<const int, int>, std::less<int> > >::value, "");
static_assert(std::is_empty<std::__map_value_compare<int, std::pair<const int, int>, AlignedLess> >::value, "");
static_assert(!std::is_empty<std::__map_value_compare<int, std::pair<const int, int>, FinalLess> >::value, "");
static_assert(!std::is_empty<std::__map_value_compare<int, std::pair<const int, int>, NonEmptyLess> >::value, "");

#if __SIZE_WIDTH__ == 64
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 32, "");
#  else
static_assert(sizeof(user_struct) == 24, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 8, "");

static_assert(sizeof(map_alloc<int, std::allocator<std::pair<const int, int> > >) == 24, "");
static_assert(sizeof(map_alloc<int, min_allocator<std::pair<const int, int> > >) == 24, "");
static_assert(sizeof(map_alloc<int, test_allocator<std::pair<const int, int> > >) == 40, "");
static_assert(sizeof(map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 6, "");
static_assert(sizeof(map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 8, "");

static_assert(sizeof(map_alloc<char, std::allocator<std::pair<const char, char> > >) == 24, "");
static_assert(sizeof(map_alloc<char, min_allocator<std::pair<const char, char> > >) == 24, "");
static_assert(sizeof(map_alloc<char, test_allocator<std::pair<const char, char> > >) == 40, "");
static_assert(sizeof(map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 6, "");
static_assert(sizeof(map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 8, "");

static_assert(TEST_ALIGNOF(map_alloc<int, std::allocator<std::pair<const int, int> > >) == 8, "");
static_assert(TEST_ALIGNOF(map_alloc<int, min_allocator<std::pair<const int, int> > >) == 8, "");
static_assert(TEST_ALIGNOF(map_alloc<int, test_allocator<std::pair<const int, int> > >) == 8, "");
static_assert(TEST_ALIGNOF(map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 2, "");
static_assert(TEST_ALIGNOF(map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 2, "");

static_assert(TEST_ALIGNOF(map_alloc<char, std::allocator<std::pair<const char, char> > >) == 8, "");
static_assert(TEST_ALIGNOF(map_alloc<char, min_allocator<std::pair<const char, char> > >) == 8, "");
static_assert(TEST_ALIGNOF(map_alloc<char, test_allocator<std::pair<const char, char> > >) == 8, "");
static_assert(TEST_ALIGNOF(map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 2, "");
static_assert(TEST_ALIGNOF(map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 2, "");

static_assert(sizeof(std::map<int, int, AlignedLess>) == 64, "");
static_assert(sizeof(std::map<int, int, FinalLess>) == 32, "");
static_assert(sizeof(std::map<int, int, NonEmptyLess>) == 32, "");

static_assert(TEST_ALIGNOF(std::map<int, int, AlignedLess>) == 32, "");
static_assert(TEST_ALIGNOF(std::map<int, int, FinalLess>) == 8, "");
static_assert(TEST_ALIGNOF(std::map<int, int, NonEmptyLess>) == 8, "");

#elif __SIZE_WIDTH__ == 32
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 16, "");
#  else
static_assert(sizeof(user_struct) == 12, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 4, "");

static_assert(sizeof(map_alloc<int, std::allocator<std::pair<const int, int> > >) == 12, "");
static_assert(sizeof(map_alloc<int, min_allocator<std::pair<const int, int> > >) == 12, "");
static_assert(sizeof(map_alloc<int, test_allocator<std::pair<const int, int> > >) == 24, "");
static_assert(sizeof(map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 6, "");
static_assert(sizeof(map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 8, "");

static_assert(sizeof(map_alloc<char, std::allocator<std::pair<const char, char> > >) == 12, "");
static_assert(sizeof(map_alloc<char, min_allocator<std::pair<const char, char> > >) == 12, "");
static_assert(sizeof(map_alloc<char, test_allocator<std::pair<const char, char> > >) == 24, "");
static_assert(sizeof(map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 6, "");
static_assert(sizeof(map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 8, "");

static_assert(TEST_ALIGNOF(map_alloc<int, std::allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(map_alloc<int, min_allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(map_alloc<int, test_allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 2, "");
static_assert(TEST_ALIGNOF(map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 2, "");

static_assert(TEST_ALIGNOF(map_alloc<char, std::allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(map_alloc<char, min_allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(map_alloc<char, test_allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 2, "");
static_assert(TEST_ALIGNOF(map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 2, "");

static_assert(sizeof(std::map<int, int, AlignedLess>) == 64, "");
static_assert(sizeof(std::map<int, int, FinalLess>) == 16, "");
static_assert(sizeof(std::map<int, int, NonEmptyLess>) == 20, "");

static_assert(TEST_ALIGNOF(std::map<int, int, AlignedLess>) == 32, "");
static_assert(TEST_ALIGNOF(std::map<int, int, FinalLess>) == 4, "");
static_assert(TEST_ALIGNOF(std::map<int, int, NonEmptyLess>) == 4, "");

#else
#  error std::size_t has an unexpected size
#endif
