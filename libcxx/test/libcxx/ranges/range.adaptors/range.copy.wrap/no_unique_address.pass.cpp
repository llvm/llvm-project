//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// This test ensures that <copyable-box> behaves correctly when it holds an empty type.

#include <ranges>

#include <cassert>
#include <utility>

bool copied = false;
bool moved = false;

struct Empty {
  Empty() noexcept { }
  Empty(Empty const&) noexcept { copied = true; }
  Empty(Empty&&) noexcept { moved = true; }
  Empty& operator=(Empty const&) = delete;
  Empty& operator=(Empty&&) = delete;
};

using Box = std::ranges::__copyable_box<Empty>;

struct Inherit : Box { };

struct Hold : Box {
  [[no_unique_address]] Inherit member;
};

int main(int, char**) {
  Hold box;

  Box& base = static_cast<Box&>(box);
  Box& member = static_cast<Box&>(box.member);

  // Despite [[no_unique_address]], the two objects have the same type so they
  // can't share the same address.
  assert(&base != &member);

  // Make sure that we do perform the copy-construction, which wouldn't be the
  // case if the two <copyable-box>s had the same address.
  base = member;
  assert(copied);

  base = std::move(member);
  assert(moved);

  return 0;
}
