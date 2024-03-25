//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__type_traits/is_trivially_relocatable.h>
#include <memory>
#include <string>

#include "constexpr_char_traits.h"
#include "test_allocator.h"

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
#ifndef _MSC_VER
static_assert(std::__libcpp_is_trivially_relocatable<MoveOnlyTriviallyCopyable>::value, "");
#else
static_assert(!std::__libcpp_is_trivially_relocatable<MoveOnlyTriviallyCopyable>::value, "");
#endif
// standard library types
// ----------------------

// basic_string
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

// TODO: Mark all the trivially relocatable STL types as such
