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
#include <unordered_map>

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
using unordered_map_alloc = std::unordered_map<T, T, std::hash<T>, std::equal_to<T>, Alloc>;

struct user_struct {
  unordered_map_alloc<int, common_base_allocator<std::pair<const int, int> > > v;
  [[no_unique_address]] common_base_allocator<int> a;
};

struct TEST_ALIGNAS(32) AlignedHash {};
struct FinalHash final {};
struct NonEmptyHash final {
  int i;
  char c;
};
struct UnalignedEqualTo {};
struct FinalEqualTo final {};
struct NonEmptyEqualTo {
  int i;
  char c;
};

static_assert(std::is_empty<std::__unordered_map_hasher<int, std::pair<const int, int>, std::hash<int>, int> >::value,
              "");
static_assert(std::is_empty<std::__unordered_map_hasher<int, std::pair<const int, int>, AlignedHash, int> >::value, "");
static_assert(!std::is_empty<std::__unordered_map_hasher<int, std::pair<const int, int>, FinalHash, int> >::value, "");
static_assert(!std::is_empty<std::__unordered_map_hasher<int, std::pair<const int, int>, NonEmptyHash, int> >::value,
              "");

static_assert(std::is_empty<std::__unordered_map_equal<int, std::pair<const int, int>, std::hash<int>, int> >::value,
              "");
static_assert(std::is_empty<std::__unordered_map_equal<int, std::pair<const int, int>, AlignedHash, int> >::value, "");
static_assert(!std::is_empty<std::__unordered_map_equal<int, std::pair<const int, int>, FinalHash, int> >::value, "");
static_assert(!std::is_empty<std::__unordered_map_equal<int, std::pair<const int, int>, NonEmptyHash, int> >::value,
              "");

#if __SIZE_WIDTH__ == 64
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 48, "");
#  else
static_assert(sizeof(user_struct) == 40, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 8, "");

static_assert(sizeof(unordered_map_alloc<int, std::allocator<std::pair<const int, int> > >) == 40, "");
static_assert(sizeof(unordered_map_alloc<int, min_allocator<std::pair<const int, int> > >) == 40, "");
static_assert(sizeof(unordered_map_alloc<int, test_allocator<std::pair<const int, int> > >) == 64, "");
static_assert(sizeof(unordered_map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 12, "");
static_assert(sizeof(unordered_map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 16, "");

static_assert(sizeof(unordered_map_alloc<char, std::allocator<std::pair<const char, char> > >) == 40, "");
static_assert(sizeof(unordered_map_alloc<char, min_allocator<std::pair<const char, char> > >) == 40, "");
static_assert(sizeof(unordered_map_alloc<char, test_allocator<std::pair<const char, char> > >) == 64, "");
static_assert(sizeof(unordered_map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 12, "");
static_assert(sizeof(unordered_map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 16, "");

static_assert(TEST_ALIGNOF(unordered_map_alloc<int, std::allocator<std::pair<const int, int> > >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, min_allocator<std::pair<const int, int> > >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, test_allocator<std::pair<const int, int> > >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 4, "");

static_assert(TEST_ALIGNOF(unordered_map_alloc<char, std::allocator<std::pair<const char, char> > >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, min_allocator<std::pair<const char, char> > >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, test_allocator<std::pair<const char, char> > >) == 8, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 4,
              "");

#ifdef TEST_COMPILER_GCC
static_assert(sizeof(std::unordered_map<int, int, FinalHash, UnalignedEqualTo>) == 40, "");
#else
static_assert(sizeof(std::unordered_map<int, int, FinalHash, UnalignedEqualTo>) == 48, "");
#endif
static_assert(sizeof(std::unordered_map<int, int, FinalHash, FinalEqualTo>) == 48, "");
static_assert(sizeof(std::unordered_map<int, int, NonEmptyHash, FinalEqualTo>) == 48, "");
static_assert(sizeof(std::unordered_map<int, int, NonEmptyHash, NonEmptyEqualTo>) == 56, "");

static_assert(TEST_ALIGNOF(std::unordered_map<int, int, FinalHash, UnalignedEqualTo>) == 8, "");
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, FinalHash, FinalEqualTo>) == 8, "");
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, NonEmptyHash, FinalEqualTo>) == 8, "");
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, NonEmptyHash, NonEmptyEqualTo>) == 8, "");

// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(std::unordered_map<int, int, AlignedHash, UnalignedEqualTo>) == 64, "");
#  else
static_assert(sizeof(std::unordered_map<int, int, AlignedHash, UnalignedEqualTo>) == 96, "");
#  endif
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, AlignedHash, UnalignedEqualTo>) == 32, "");

#elif __SIZE_WIDTH__ == 32
// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(user_struct) == 24, "");
#  else
static_assert(sizeof(user_struct) == 20, "");
#  endif
static_assert(TEST_ALIGNOF(user_struct) == 4, "");

static_assert(sizeof(unordered_map_alloc<int, std::allocator<std::pair<const int, int> > >) == 20, "");
static_assert(sizeof(unordered_map_alloc<int, min_allocator<std::pair<const int, int> > >) == 20, "");
static_assert(sizeof(unordered_map_alloc<int, test_allocator<std::pair<const int, int> > >) == 44, "");
static_assert(sizeof(unordered_map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 12, "");
static_assert(sizeof(unordered_map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 16, "");

static_assert(sizeof(unordered_map_alloc<char, std::allocator<std::pair<const char, char> > >) == 20, "");
static_assert(sizeof(unordered_map_alloc<char, min_allocator<std::pair<const char, char> > >) == 20, "");
static_assert(sizeof(unordered_map_alloc<char, test_allocator<std::pair<const char, char> > >) == 44, "");
static_assert(sizeof(unordered_map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 12, "");
static_assert(sizeof(unordered_map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 16, "");

static_assert(TEST_ALIGNOF(unordered_map_alloc<int, std::allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, min_allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, test_allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, small_iter_allocator<std::pair<const int, int> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<int, final_small_iter_allocator<std::pair<const int, int> > >) == 4, "");

static_assert(TEST_ALIGNOF(unordered_map_alloc<char, std::allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, min_allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, test_allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, small_iter_allocator<std::pair<const char, char> > >) == 4, "");
static_assert(TEST_ALIGNOF(unordered_map_alloc<char, final_small_iter_allocator<std::pair<const char, char> > >) == 4,
              "");

static_assert(sizeof(std::unordered_map<int, int, FinalHash, UnalignedEqualTo>) == 24, "");
static_assert(sizeof(std::unordered_map<int, int, FinalHash, FinalEqualTo>) == 24, "");
static_assert(sizeof(std::unordered_map<int, int, NonEmptyHash, FinalEqualTo>) == 28, "");
static_assert(sizeof(std::unordered_map<int, int, NonEmptyHash, NonEmptyEqualTo>) == 32, "");

static_assert(TEST_ALIGNOF(std::unordered_map<int, int, FinalHash, UnalignedEqualTo>) == 4, "");
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, FinalHash, FinalEqualTo>) == 4, "");
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, NonEmptyHash, FinalEqualTo>) == 4, "");
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, NonEmptyHash, NonEmptyEqualTo>) == 4, "");

// TODO: Fix the ABI for GCC as well once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=121637 is fixed
#  ifdef TEST_COMPILER_GCC
static_assert(sizeof(std::unordered_map<int, int, AlignedHash, UnalignedEqualTo>) == 64);
#  else
static_assert(sizeof(std::unordered_map<int, int, AlignedHash, UnalignedEqualTo>) == 96);
#  endif
static_assert(TEST_ALIGNOF(std::unordered_map<int, int, AlignedHash, UnalignedEqualTo>) == 32);

#else
#  error std::size_t has an unexpected size
#endif
