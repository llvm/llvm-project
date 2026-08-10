//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__type_traits/is_trivially_relocatable.h>
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

static_assert(std::__is_trivially_relocatable_v<char>, "");
static_assert(std::__is_trivially_relocatable_v<int>, "");
static_assert(std::__is_trivially_relocatable_v<double>, "");

struct Empty {};
static_assert(std::__is_trivially_relocatable_v<Empty>, "");

struct TriviallyCopyable {
  char c;
  int i;
  Empty s;
};
static_assert(std::__is_trivially_relocatable_v<TriviallyCopyable>, "");

struct NotTriviallyCopyable {
  NotTriviallyCopyable(const NotTriviallyCopyable&);
  ~NotTriviallyCopyable();
};
static_assert(!std::__is_trivially_relocatable_v<NotTriviallyCopyable>, "");

struct MoveOnlyTriviallyCopyable {
  MoveOnlyTriviallyCopyable(const MoveOnlyTriviallyCopyable&)            = delete;
  MoveOnlyTriviallyCopyable& operator=(const MoveOnlyTriviallyCopyable&) = delete;
  MoveOnlyTriviallyCopyable(MoveOnlyTriviallyCopyable&&)                 = default;
  MoveOnlyTriviallyCopyable& operator=(MoveOnlyTriviallyCopyable&&)      = default;
};
static_assert(std::__is_trivially_relocatable_v<MoveOnlyTriviallyCopyable>, "");

struct NonTrivialMoveConstructor {
  NonTrivialMoveConstructor(NonTrivialMoveConstructor&&);
};
static_assert(!std::__is_trivially_relocatable_v<NonTrivialMoveConstructor>, "");

struct NonTrivialDestructor {
  ~NonTrivialDestructor() {}
};
static_assert(!std::__is_trivially_relocatable_v<NonTrivialDestructor>, "");

// library-internal types
// ----------------------

// __split_buffer
static_assert(std::__is_trivially_relocatable_v<
                  std::__split_buffer<int, std::allocator<int>, std::__split_buffer_pointer_layout> >,
              "");
static_assert(std::__is_trivially_relocatable_v<std::__split_buffer<NotTriviallyCopyable,
                                                                    std::allocator<NotTriviallyCopyable>,
                                                                    std::__split_buffer_pointer_layout> >,
              "");
static_assert(!std::__is_trivially_relocatable_v<
                  std::__split_buffer<int, test_allocator<int>, std::__split_buffer_pointer_layout > >,
              "");

static_assert(
    std::__is_trivially_relocatable_v< std::__split_buffer<int, std::allocator<int>, std::__split_buffer_size_layout> >,
    "");
static_assert(std::__is_trivially_relocatable_v<std::__split_buffer<NotTriviallyCopyable,
                                                                    std::allocator<NotTriviallyCopyable>,
                                                                    std::__split_buffer_size_layout> >,
              "");
static_assert(!std::__is_trivially_relocatable_v<
                  std::__split_buffer<int, test_allocator<int>, std::__split_buffer_size_layout > >,
              "");

// standard library types
// ----------------------

// array
static_assert(std::__is_trivially_relocatable_v<std::array<int, 0> >, "");
static_assert(std::__is_trivially_relocatable_v<std::array<NotTriviallyCopyable, 0> >, "");
static_assert(std::__is_trivially_relocatable_v<std::array<std::unique_ptr<int>, 0> >, "");

static_assert(std::__is_trivially_relocatable_v<std::array<int, 1> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::array<NotTriviallyCopyable, 1> >, "");
static_assert(std::__is_trivially_relocatable_v<std::array<std::unique_ptr<int>, 1> >, "");

// basic_string
#if !_LIBCPP_ENABLE_ASAN_CONTAINER_CHECKS_FOR_STRING
struct MyChar {
  char c;
};
template <class T>
struct NotTriviallyRelocatableCharTraits : constexpr_char_traits<T> {
  NotTriviallyRelocatableCharTraits(const NotTriviallyRelocatableCharTraits&);
  NotTriviallyRelocatableCharTraits& operator=(const NotTriviallyRelocatableCharTraits&);
  ~NotTriviallyRelocatableCharTraits();
};

static_assert(
    std::__is_trivially_relocatable_v< std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, "");
static_assert(std::__is_trivially_relocatable_v<
                  std::basic_string<char, NotTriviallyRelocatableCharTraits<char>, std::allocator<char> > >,
              "");
static_assert(std::__is_trivially_relocatable_v<
                  std::basic_string<MyChar, constexpr_char_traits<MyChar>, std::allocator<MyChar> > >,
              "");
static_assert(std::__is_trivially_relocatable_v<
                  std::basic_string<MyChar, NotTriviallyRelocatableCharTraits<MyChar>, std::allocator<MyChar> > >,
              "");
static_assert(
    !std::__is_trivially_relocatable_v< std::basic_string<char, std::char_traits<char>, test_allocator<char> > >, "");
static_assert(!std::__is_trivially_relocatable_v<
                  std::basic_string<MyChar, NotTriviallyRelocatableCharTraits<MyChar>, test_allocator<MyChar> > >,
              "");
#endif

// deque
static_assert(std::__is_trivially_relocatable_v<std::deque<int> >, "");
static_assert(std::__is_trivially_relocatable_v<std::deque<NotTriviallyCopyable> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::deque<int, test_allocator<int> > >, "");

// exception_ptr
#ifndef _LIBCPP_ABI_MICROSOFT // FIXME: Is this also the case on windows?
static_assert(std::__is_trivially_relocatable_v<std::exception_ptr>, "");
#endif

// expected
#if TEST_STD_VER >= 23
static_assert(std::__is_trivially_relocatable_v<std::expected<int, int> >);
static_assert(std::__is_trivially_relocatable_v<std::expected<std::unique_ptr<int>, int>>);
static_assert(std::__is_trivially_relocatable_v<std::expected<int, std::unique_ptr<int>>>);
static_assert(std::__is_trivially_relocatable_v<std::expected<std::unique_ptr<int>, std::unique_ptr<int>>>);

static_assert(!std::__is_trivially_relocatable_v<std::expected<int, NotTriviallyCopyable>>);
static_assert(!std::__is_trivially_relocatable_v<std::expected<NotTriviallyCopyable, int>>);
static_assert(!std::__is_trivially_relocatable_v<std::expected<NotTriviallyCopyable, NotTriviallyCopyable>>);
#endif

// locale
#ifndef TEST_HAS_NO_LOCALIZATION
static_assert(std::__is_trivially_relocatable_v<std::locale>, "");
#endif

// optional
#if TEST_STD_VER >= 17
static_assert(std::__is_trivially_relocatable_v<std::optional<int>>, "");
static_assert(!std::__is_trivially_relocatable_v<std::optional<NotTriviallyCopyable>>, "");
static_assert(std::__is_trivially_relocatable_v<std::optional<std::unique_ptr<int>>>, "");
#endif // TEST_STD_VER >= 17

// pair
static_assert(std::__is_trivially_relocatable_v<std::pair<int, int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::pair<NotTriviallyCopyable, int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::pair<int, NotTriviallyCopyable> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::pair<NotTriviallyCopyable, NotTriviallyCopyable> >, "");
static_assert(std::__is_trivially_relocatable_v<std::pair<std::unique_ptr<int>, std::unique_ptr<int> > >, "");

// shared_ptr
static_assert(std::__is_trivially_relocatable_v<std::shared_ptr<NotTriviallyCopyable> >, "");

// tuple
#if TEST_STD_VER >= 11
static_assert(std::__is_trivially_relocatable_v<std::tuple<> >, "");

static_assert(std::__is_trivially_relocatable_v<std::tuple<int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::tuple<NotTriviallyCopyable> >, "");
static_assert(std::__is_trivially_relocatable_v<std::tuple<std::unique_ptr<int> > >, "");

static_assert(std::__is_trivially_relocatable_v<std::tuple<int, int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::tuple<NotTriviallyCopyable, int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::tuple<int, NotTriviallyCopyable> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::tuple<NotTriviallyCopyable, NotTriviallyCopyable> >, "");
static_assert(std::__is_trivially_relocatable_v<std::tuple<std::unique_ptr<int>, std::unique_ptr<int> > >, "");
#endif // TEST_STD_VER >= 11

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

static_assert(std::__is_trivially_relocatable_v<std::unique_ptr<int> >, "");
static_assert(std::__is_trivially_relocatable_v<std::unique_ptr<NotTriviallyCopyable> >, "");
static_assert(std::__is_trivially_relocatable_v<std::unique_ptr<int[]> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::unique_ptr<int, NotTriviallyRelocatableDeleter> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::unique_ptr<int[], NotTriviallyRelocatableDeleter> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::unique_ptr<int, NotTriviallyRelocatablePointer> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::unique_ptr<int[], NotTriviallyRelocatablePointer> >, "");

// variant
#if TEST_STD_VER >= 17
static_assert(std::__is_trivially_relocatable_v<std::variant<int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::variant<NotTriviallyCopyable> >, "");
static_assert(std::__is_trivially_relocatable_v<std::variant<std::unique_ptr<int> > >, "");

static_assert(std::__is_trivially_relocatable_v<std::variant<int, int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::variant<NotTriviallyCopyable, int> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::variant<int, NotTriviallyCopyable> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::variant<NotTriviallyCopyable, NotTriviallyCopyable> >, "");
static_assert(std::__is_trivially_relocatable_v<std::variant<std::unique_ptr<int>, std::unique_ptr<int> > >, "");
#endif // TEST_STD_VER >= 17

// vector
static_assert(std::__is_trivially_relocatable_v<std::vector<int> >, "");
static_assert(std::__is_trivially_relocatable_v<std::vector<NotTriviallyCopyable> >, "");
static_assert(!std::__is_trivially_relocatable_v<std::vector<int, test_allocator<int> > >, "");

// weak_ptr
static_assert(std::__is_trivially_relocatable_v<std::weak_ptr<NotTriviallyCopyable> >, "");

// TODO: Mark all the trivially relocatable STL types as such
