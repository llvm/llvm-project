//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__type_traits/is_replaceable.h>
#include <array>
#include <deque>
#include <exception>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "constexpr_char_traits.h"
#include "test_allocator.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <locale>
#endif

template <class T>
struct NonPropagatingStatefulMoveAssignAlloc : std::allocator<T> {
  using propagate_on_container_move_assignment = std::false_type;
  using is_always_equal                        = std::false_type;
  template <class U>
  struct rebind {
    using other = NonPropagatingStatefulMoveAssignAlloc<U>;
  };
};

template <class T>
struct NonPropagatingStatefulCopyAssignAlloc : std::allocator<T> {
  using propagate_on_container_copy_assignment = std::false_type;
  using is_always_equal                        = std::false_type;
  template <class U>
  struct rebind {
    using other = NonPropagatingStatefulCopyAssignAlloc<U>;
  };
};

template <class T>
struct NonPropagatingStatelessMoveAssignAlloc : std::allocator<T> {
  using propagate_on_container_move_assignment = std::false_type;
  using is_always_equal                        = std::true_type;
  template <class U>
  struct rebind {
    using other = NonPropagatingStatelessMoveAssignAlloc<U>;
  };
};

template <class T>
struct NonPropagatingStatelessCopyAssignAlloc : std::allocator<T> {
  using propagate_on_container_copy_assignment = std::false_type;
  using is_always_equal                        = std::true_type;
  template <class U>
  struct rebind {
    using other = NonPropagatingStatelessCopyAssignAlloc<U>;
  };
};

template <class T>
struct NonReplaceableStatelessAlloc : std::allocator<T> {
  // Ensure that we don't consider an allocator that is a member of a container to be
  // replaceable if it's not replaceable, even if it always compares equal and always propagates.
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_copy_assignment = std::true_type;
  using is_always_equal                        = std::true_type;
  NonReplaceableStatelessAlloc()               = default;
  NonReplaceableStatelessAlloc(NonReplaceableStatelessAlloc const&) {}
  NonReplaceableStatelessAlloc(NonReplaceableStatelessAlloc&&) = default;
  template <class U>
  struct rebind {
    using other = NonReplaceableStatelessAlloc<U>;
  };
};
static_assert(!std::__is_replaceable<NonReplaceableStatelessAlloc<int> >::value, "");

static_assert(!std::__is_replaceable<test_allocator<char> >::value, ""); // we use that property below

struct Empty {};
static_assert(std::__is_replaceable<char>::value, "");
static_assert(std::__is_replaceable<int>::value, "");
static_assert(std::__is_replaceable<double>::value, "");
static_assert(std::__is_replaceable<Empty>::value, "");

struct TriviallyCopyable {
  char c;
  int i;
  Empty s;
};
static_assert(std::__is_replaceable<TriviallyCopyable>::value, "");

struct NotTriviallyCopyable {
  NotTriviallyCopyable(const NotTriviallyCopyable&);
  ~NotTriviallyCopyable();
};
static_assert(!std::__is_replaceable<NotTriviallyCopyable>::value, "");

struct MoveOnlyTriviallyCopyable {
  MoveOnlyTriviallyCopyable(const MoveOnlyTriviallyCopyable&)            = delete;
  MoveOnlyTriviallyCopyable& operator=(const MoveOnlyTriviallyCopyable&) = delete;
  MoveOnlyTriviallyCopyable(MoveOnlyTriviallyCopyable&&)                 = default;
  MoveOnlyTriviallyCopyable& operator=(MoveOnlyTriviallyCopyable&&)      = default;
};
static_assert(std::__is_replaceable<MoveOnlyTriviallyCopyable>::value, "");

struct CustomCopyAssignment {
  CustomCopyAssignment(const CustomCopyAssignment&) = default;
  CustomCopyAssignment(CustomCopyAssignment&&)      = default;
  CustomCopyAssignment& operator=(const CustomCopyAssignment&);
  CustomCopyAssignment& operator=(CustomCopyAssignment&&) = default;
};
static_assert(!std::__is_replaceable<CustomCopyAssignment>::value, "");

struct CustomMoveAssignment {
  CustomMoveAssignment(const CustomMoveAssignment&)            = default;
  CustomMoveAssignment(CustomMoveAssignment&&)                 = default;
  CustomMoveAssignment& operator=(const CustomMoveAssignment&) = default;
  CustomMoveAssignment& operator=(CustomMoveAssignment&&);
};
static_assert(!std::__is_replaceable<CustomMoveAssignment>::value, "");

// library-internal types
// ----------------------

// __split_buffer
static_assert(std::__is_replaceable<std::__split_buffer<int> >::value, "");
static_assert(std::__is_replaceable<std::__split_buffer<NotTriviallyCopyable> >::value, "");
static_assert(!std::__is_replaceable<std::__split_buffer<int, NonPropagatingStatefulCopyAssignAlloc<int> > >::value,
              "");
static_assert(!std::__is_replaceable<std::__split_buffer<int, NonPropagatingStatefulMoveAssignAlloc<int> > >::value,
              "");
static_assert(std::__is_replaceable<std::__split_buffer<int, NonPropagatingStatelessCopyAssignAlloc<int> > >::value,
              "");
static_assert(std::__is_replaceable<std::__split_buffer<int, NonPropagatingStatelessMoveAssignAlloc<int> > >::value,
              "");

// standard library types
// ----------------------

// array
static_assert(std::__is_replaceable<std::array<int, 0> >::value, "");
static_assert(std::__is_replaceable<std::array<NotTriviallyCopyable, 0> >::value, "");
static_assert(std::__is_replaceable<std::array<std::unique_ptr<int>, 0> >::value, "");

static_assert(std::__is_replaceable<std::array<int, 1> >::value, "");
static_assert(!std::__is_replaceable<std::array<NotTriviallyCopyable, 1> >::value, "");
static_assert(std::__is_replaceable<std::array<std::unique_ptr<int>, 1> >::value, "");

// basic_string
struct MyChar {
  char c;
};
template <class T>
struct NotReplaceableCharTraits : constexpr_char_traits<T> {
  NotReplaceableCharTraits(const NotReplaceableCharTraits&);
  NotReplaceableCharTraits& operator=(const NotReplaceableCharTraits&);
  ~NotReplaceableCharTraits();
};

static_assert(std::__is_replaceable<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >::value,
              "");
static_assert(
    std::__is_replaceable<std::basic_string<char, NotReplaceableCharTraits<char>, std::allocator<char> > >::value, "");
static_assert(
    std::__is_replaceable<std::basic_string<MyChar, constexpr_char_traits<MyChar>, std::allocator<MyChar> > >::value,
    "");
static_assert(!std::__is_replaceable<std::basic_string<char, std::char_traits<char>, test_allocator<char> > >::value,
              "");
static_assert(!std::__is_replaceable<
                  std::basic_string<char, std::char_traits<char>, NonReplaceableStatelessAlloc<char> > >::value,
              "");
static_assert(std::__is_replaceable<
                  std::basic_string<MyChar, NotReplaceableCharTraits<MyChar>, std::allocator<MyChar> > >::value,
              "");
static_assert(
    !std::__is_replaceable<
        std::basic_string<char, std::char_traits<char>, NonPropagatingStatefulCopyAssignAlloc<char> > >::value,
    "");
static_assert(
    !std::__is_replaceable<
        std::basic_string<char, std::char_traits<char>, NonPropagatingStatefulMoveAssignAlloc<char> > >::value,
    "");
static_assert(
    std::__is_replaceable<
        std::basic_string<char, std::char_traits<char>, NonPropagatingStatelessCopyAssignAlloc<char> > >::value,
    "");
static_assert(
    std::__is_replaceable<
        std::basic_string<char, std::char_traits<char>, NonPropagatingStatelessMoveAssignAlloc<char> > >::value,
    "");

// deque
static_assert(std::__is_replaceable<std::deque<int> >::value, "");
static_assert(std::__is_replaceable<std::deque<NotTriviallyCopyable> >::value, "");
static_assert(!std::__is_replaceable<std::deque<int, test_allocator<int> > >::value, "");
static_assert(!std::__is_replaceable<std::deque<int, NonReplaceableStatelessAlloc<int> > >::value, "");
static_assert(!std::__is_replaceable<std::deque<int, NonPropagatingStatefulCopyAssignAlloc<int> > >::value, "");
static_assert(!std::__is_replaceable<std::deque<int, NonPropagatingStatefulMoveAssignAlloc<int> > >::value, "");
static_assert(std::__is_replaceable<std::deque<int, NonPropagatingStatelessCopyAssignAlloc<int> > >::value, "");
static_assert(std::__is_replaceable<std::deque<int, NonPropagatingStatelessMoveAssignAlloc<int> > >::value, "");

// exception_ptr
#ifndef _LIBCPP_ABI_MICROSOFT
static_assert(std::__is_replaceable<std::exception_ptr>::value, "");
#endif

// expected
#if TEST_STD_VER >= 23
static_assert(std::__is_replaceable<std::expected<int, int> >::value);
static_assert(!std::__is_replaceable<std::expected<CustomCopyAssignment, int>>::value);
static_assert(!std::__is_replaceable<std::expected<int, CustomCopyAssignment>>::value);
static_assert(!std::__is_replaceable<std::expected<CustomCopyAssignment, CustomCopyAssignment>>::value);
#endif

// locale
#ifndef TEST_HAS_NO_LOCALIZATION
static_assert(std::__is_replaceable<std::locale>::value, "");
#endif

// optional
#if TEST_STD_VER >= 17
static_assert(std::__is_replaceable<std::optional<int>>::value, "");
static_assert(!std::__is_replaceable<std::optional<CustomCopyAssignment>>::value, "");
#endif

// pair
static_assert(std::__is_replaceable<std::pair<int, int> >::value, "");
static_assert(!std::__is_replaceable<std::pair<CustomCopyAssignment, int> >::value, "");
static_assert(!std::__is_replaceable<std::pair<int, CustomCopyAssignment> >::value, "");
static_assert(!std::__is_replaceable<std::pair<CustomCopyAssignment, CustomCopyAssignment> >::value, "");

// shared_ptr
static_assert(std::__is_replaceable<std::shared_ptr<int> >::value, "");

// tuple
#if TEST_STD_VER >= 11
static_assert(std::__is_replaceable<std::tuple<> >::value, "");

static_assert(std::__is_replaceable<std::tuple<int> >::value, "");
static_assert(!std::__is_replaceable<std::tuple<CustomCopyAssignment> >::value, "");

static_assert(std::__is_replaceable<std::tuple<int, int> >::value, "");
static_assert(!std::__is_replaceable<std::tuple<CustomCopyAssignment, int> >::value, "");
static_assert(!std::__is_replaceable<std::tuple<int, CustomCopyAssignment> >::value, "");
static_assert(!std::__is_replaceable<std::tuple<CustomCopyAssignment, CustomCopyAssignment> >::value, "");
#endif // TEST_STD_VER >= 11

// unique_ptr
struct NonReplaceableDeleter {
  NonReplaceableDeleter(const NonReplaceableDeleter&);
  NonReplaceableDeleter& operator=(const NonReplaceableDeleter&);
  ~NonReplaceableDeleter();

  template <class T>
  void operator()(T*);
};

struct NonReplaceablePointer {
  struct pointer {
    pointer(const pointer&);
    pointer& operator=(const pointer&);
    ~pointer();
  };

  template <class T>
  void operator()(T*);
};

static_assert(std::__is_replaceable<std::unique_ptr<int> >::value, "");
static_assert(std::__is_replaceable<std::unique_ptr<CustomCopyAssignment> >::value, "");
static_assert(std::__is_replaceable<std::unique_ptr<int[]> >::value, "");
static_assert(!std::__is_replaceable<std::unique_ptr<int, NonReplaceableDeleter> >::value, "");
static_assert(!std::__is_replaceable<std::unique_ptr<int[], NonReplaceableDeleter> >::value, "");
static_assert(!std::__is_replaceable<std::unique_ptr<int, NonReplaceablePointer> >::value, "");
static_assert(!std::__is_replaceable<std::unique_ptr<int[], NonReplaceablePointer> >::value, "");

// variant
#if TEST_STD_VER >= 17
static_assert(std::__is_replaceable<std::variant<int> >::value, "");
static_assert(!std::__is_replaceable<std::variant<CustomCopyAssignment> >::value, "");

static_assert(std::__is_replaceable<std::variant<int, int> >::value, "");
static_assert(!std::__is_replaceable<std::variant<CustomCopyAssignment, int> >::value, "");
static_assert(!std::__is_replaceable<std::variant<int, CustomCopyAssignment> >::value, "");
static_assert(!std::__is_replaceable<std::variant<CustomCopyAssignment, CustomCopyAssignment> >::value, "");
#endif // TEST_STD_VER >= 17

// vector
static_assert(std::__is_replaceable<std::vector<int> >::value, "");
static_assert(std::__is_replaceable<std::vector<CustomCopyAssignment> >::value, "");
static_assert(!std::__is_replaceable<std::vector<int, test_allocator<int> > >::value, "");
static_assert(!std::__is_replaceable<std::vector<int, NonReplaceableStatelessAlloc<int> > >::value, "");
static_assert(!std::__is_replaceable<std::vector<int, NonPropagatingStatefulCopyAssignAlloc<int> > >::value, "");
static_assert(!std::__is_replaceable<std::vector<int, NonPropagatingStatefulMoveAssignAlloc<int> > >::value, "");
static_assert(std::__is_replaceable<std::vector<int, NonPropagatingStatelessCopyAssignAlloc<int> > >::value, "");
static_assert(std::__is_replaceable<std::vector<int, NonPropagatingStatelessMoveAssignAlloc<int> > >::value, "");

// weak_ptr
static_assert(std::__is_replaceable<std::weak_ptr<CustomCopyAssignment> >::value, "");

// TODO: Mark all the replaceable STL types as such
