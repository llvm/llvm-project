//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__cxx03/__type_traits/is_trivially_relocatable.h>
#include <array>
#include <deque>
#include <exception>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "constexpr_char_traits.h"
#include "test_allocator.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <locale>
#endif

static_assert(std::__libcpp_is_trivially_relocatable<char>::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<int>::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<double>::value, "");

struct Empty {};
static_assert(std::__libcpp_is_trivially_relocatable<Empty>::value, "");

struct TriviallyCopyable {
  char c;
  int i;
  Empty s;
};
static_assert(std::__libcpp_is_trivially_relocatable<TriviallyCopyable>::value, "");

struct NotTriviallyCopyable {
  NotTriviallyCopyable(const NotTriviallyCopyable&);
  ~NotTriviallyCopyable();
};
static_assert(!std::__libcpp_is_trivially_relocatable<NotTriviallyCopyable>::value, "");

struct MoveOnlyTriviallyCopyable {
  MoveOnlyTriviallyCopyable(const MoveOnlyTriviallyCopyable&)            = delete;
  MoveOnlyTriviallyCopyable& operator=(const MoveOnlyTriviallyCopyable&) = delete;
  MoveOnlyTriviallyCopyable(MoveOnlyTriviallyCopyable&&)                 = default;
  MoveOnlyTriviallyCopyable& operator=(MoveOnlyTriviallyCopyable&&)      = default;
};
static_assert(std::__libcpp_is_trivially_relocatable<MoveOnlyTriviallyCopyable>::value, "");

struct NonTrivialMoveConstructor {
  NonTrivialMoveConstructor(NonTrivialMoveConstructor&&);
};
static_assert(!std::__libcpp_is_trivially_relocatable<NonTrivialMoveConstructor>::value, "");

struct NonTrivialDestructor {
  ~NonTrivialDestructor() {}
};
static_assert(!std::__libcpp_is_trivially_relocatable<NonTrivialDestructor>::value, "");

// library-internal types
// ----------------------

// __split_buffer
static_assert(std::__libcpp_is_trivially_relocatable<std::__split_buffer<int> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::__split_buffer<NotTriviallyCopyable> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::__split_buffer<int, test_allocator<int> > >::value, "");

// standard library types
// ----------------------

// array
static_assert(std::__libcpp_is_trivially_relocatable<std::array<int, 0> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::array<NotTriviallyCopyable, 0> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::array<std::unique_ptr<int>, 0> >::value, "");

static_assert(std::__libcpp_is_trivially_relocatable<std::array<int, 1> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::array<NotTriviallyCopyable, 1> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::array<std::unique_ptr<int>, 1> >::value, "");

// basic_string
#if !__has_feature(address_sanitizer) || !_LIBCPP_INSTRUMENTED_WITH_ASAN
struct MyChar {
  char c;
};
template <class T>
struct NotTriviallyRelocatableCharTraits : constexpr_char_traits<T> {
  NotTriviallyRelocatableCharTraits(const NotTriviallyRelocatableCharTraits&);
  NotTriviallyRelocatableCharTraits& operator=(const NotTriviallyRelocatableCharTraits&);
  ~NotTriviallyRelocatableCharTraits();
};

static_assert(std::__libcpp_is_trivially_relocatable<
                  std::basic_string<char, std::char_traits<char>, std::allocator<char> > >::value,
              "");
static_assert(std::__libcpp_is_trivially_relocatable<
                  std::basic_string<char, NotTriviallyRelocatableCharTraits<char>, std::allocator<char> > >::value,
              "");
static_assert(std::__libcpp_is_trivially_relocatable<
                  std::basic_string<MyChar, constexpr_char_traits<MyChar>, std::allocator<MyChar> > >::value,
              "");
static_assert(
    std::__libcpp_is_trivially_relocatable<
        std::basic_string<MyChar, NotTriviallyRelocatableCharTraits<MyChar>, std::allocator<MyChar> > >::value,
    "");
static_assert(!std::__libcpp_is_trivially_relocatable<
                  std::basic_string<char, std::char_traits<char>, test_allocator<char> > >::value,
              "");
static_assert(
    !std::__libcpp_is_trivially_relocatable<
        std::basic_string<MyChar, NotTriviallyRelocatableCharTraits<MyChar>, test_allocator<MyChar> > >::value,
    "");
#endif

// deque
static_assert(std::__libcpp_is_trivially_relocatable<std::deque<int> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::deque<NotTriviallyCopyable> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::deque<int, test_allocator<int> > >::value, "");

// exception_ptr
#ifndef _LIBCPP_ABI_MICROSOFT // FIXME: Is this also the case on windows?
static_assert(std::__libcpp_is_trivially_relocatable<std::exception_ptr>::value, "");
#endif

// locale
#ifndef TEST_HAS_NO_LOCALIZATION
static_assert(std::__libcpp_is_trivially_relocatable<std::locale>::value, "");
#endif

// pair
static_assert(std::__libcpp_is_trivially_relocatable<std::pair<int, int> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::pair<NotTriviallyCopyable, int> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::pair<int, NotTriviallyCopyable> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::pair<NotTriviallyCopyable, NotTriviallyCopyable> >::value,
              "");
static_assert(std::__libcpp_is_trivially_relocatable<std::pair<std::unique_ptr<int>, std::unique_ptr<int> > >::value,
              "");

// shared_ptr
static_assert(std::__libcpp_is_trivially_relocatable<std::shared_ptr<NotTriviallyCopyable> >::value, "");

// unique_ptr
struct NotTriviallyRelocatableDeleter {
  NotTriviallyRelocatableDeleter(const NotTriviallyRelocatableDeleter&);
  NotTriviallyRelocatableDeleter& operator=(const NotTriviallyRelocatableDeleter&);
  ~NotTriviallyRelocatableDeleter();

  template <class T>
  void operator()(T*);
};

struct NotTriviallyRelocatablePointer {
  struct pointer {
    pointer(const pointer&);
    pointer& operator=(const pointer&);
    ~pointer();
  };

  template <class T>
  void operator()(T*);
};

static_assert(std::__libcpp_is_trivially_relocatable<std::unique_ptr<int> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::unique_ptr<NotTriviallyCopyable> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::unique_ptr<int[]> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::unique_ptr<int, NotTriviallyRelocatableDeleter> >::value,
              "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::unique_ptr<int[], NotTriviallyRelocatableDeleter> >::value,
              "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::unique_ptr<int, NotTriviallyRelocatablePointer> >::value,
              "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::unique_ptr<int[], NotTriviallyRelocatablePointer> >::value,
              "");

// vector
static_assert(std::__libcpp_is_trivially_relocatable<std::vector<int> >::value, "");
static_assert(std::__libcpp_is_trivially_relocatable<std::vector<NotTriviallyCopyable> >::value, "");
static_assert(!std::__libcpp_is_trivially_relocatable<std::vector<int, test_allocator<int> > >::value, "");

// weak_ptr
static_assert(std::__libcpp_is_trivially_relocatable<std::weak_ptr<NotTriviallyCopyable> >::value, "");

// TODO: Mark all the trivially relocatable STL types as such
