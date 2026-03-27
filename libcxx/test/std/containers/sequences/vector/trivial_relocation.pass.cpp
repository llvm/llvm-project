//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Make sure we don't miscompile vector operations for types that shouldn't be considered
// trivially relocatable.

#include <vector>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct Tracker {
  std::size_t move_constructs = 0;
};

struct [[clang::trivial_abi]] Inner {
  TEST_CONSTEXPR_CXX20 explicit Inner(Tracker* tracker) : tracker_(tracker) {}
  TEST_CONSTEXPR_CXX20 Inner(const Inner& rhs) : tracker_(rhs.tracker_) { tracker_->move_constructs += 1; }
  TEST_CONSTEXPR_CXX20 Inner(Inner&& rhs) : tracker_(rhs.tracker_) { tracker_->move_constructs += 1; }
  Tracker* tracker_;
};

// Even though this type contains a trivial_abi type, it is not trivially move-constructible,
// so we should not attempt to optimize its move construction + destroy using trivial relocation.
struct NotTriviallyMovable {
  TEST_CONSTEXPR_CXX20 explicit NotTriviallyMovable(Tracker* tracker) : inner_(tracker) {}
  TEST_CONSTEXPR_CXX20 NotTriviallyMovable(NotTriviallyMovable&& other) : inner_(std::move(other.inner_)) {}
  Inner inner_;
};
static_assert(!std::is_trivially_copyable<NotTriviallyMovable>::value, "");
LIBCPP_STATIC_ASSERT(!std::__libcpp_is_trivially_relocatable<NotTriviallyMovable>::value, "");

TEST_CONSTEXPR_CXX20 bool tests() {
  Tracker track;
  std::vector<NotTriviallyMovable> v;

  // Fill the vector at its capacity, such that any subsequent push_back would require growing.
  v.reserve(5);
  std::size_t const capacity = v.capacity(); // could technically be more than 5
  while (v.size() < v.capacity()) {
    v.emplace_back(&track);
  }
  assert(track.move_constructs == 0);
  assert(v.capacity() == capacity);
  assert(v.size() == capacity);

  // Force a reallocation of the buffer + relocalization of the elements.
  // All the existing elements of the vector should be move-constructed to their new location.
  v.emplace_back(&track);
  assert(track.move_constructs == capacity);

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests());
#endif
  return 0;
}
