//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr explicit concat_view(Views... views)

#include <array>
#include <ranges>

#include "../range_adaptor_types.h"

template <class T>
void conversion_test(T);

template <class T, class... Args>
concept implicitly_constructible_from = requires(Args&&... args) { conversion_test<T>({std::move(args)...}); };

// test constructor is explicit
static_assert(
    std::constructible_from<std::ranges::concat_view<SimpleCommon, NonSimpleCommon>, SimpleCommon, NonSimpleCommon>);
static_assert(!implicitly_constructible_from<std::ranges::concat_view<SimpleCommon, NonSimpleCommon>,
                                             SimpleCommon,
                                             NonSimpleCommon>);

struct MoveAwareView : std::ranges::view_base {
  int moves                 = 0;
  constexpr MoveAwareView() = default;
  constexpr MoveAwareView(MoveAwareView&& other) : moves(other.moves + 1) { other.moves = 1; }
  constexpr MoveAwareView& operator=(MoveAwareView&& other) {
    moves       = other.moves + 1;
    other.moves = 0;
    return *this;
  }
  constexpr int* begin() { return &moves; }
  constexpr int* end() { return &moves + 1; }
};

constexpr bool test() {
  int buffer[3]  = {1, 2, 3};
  int buffer2[2] = {4, 5};

  {
    // constructor from views
    std::ranges::concat_view v(SizedRandomAccessView{buffer}, std::array<int, 1>{7});
    auto it = v.begin();
    assert(*it == 1);
    it++;
    it++;
    it++;
    assert(*it == 7);
  }

  {
    // arguments are moved
    MoveAwareView mv;
    std::ranges::concat_view v{std::move(mv), MoveAwareView{}};
    auto it = v.begin();
    assert(*it == 2);
    it++;
    assert(*it == 1);
  }

  {
    //input and forward range
    std::ranges::concat_view v(InputCommonView{buffer}, ForwardSizedView{buffer2});
    auto it = v.begin();
    assert(*it == 1);
    it++;
    assert(*it == 2);
    it++;
    assert(*it == 3);
    it++;
    assert(*it == 4);
    it++;
    assert(*it == 5);
    it++;
  }

  {
    // bidi
    std::ranges::concat_view v(BidiCommonView{buffer}, SizedBidiCommon{buffer2});
    auto it = v.begin();
    assert(*it == 1);
    it++;
    assert(*it == 2);
    it--;
    assert(*it == 1);
  }

  {
    // random access
    std::ranges::concat_view v(SimpleCommonRandomAccessSized{buffer}, SizedRandomAccessView{buffer2});
    auto it = v.begin();
    assert(*it == 1);
    assert(it[2] == 3);
    assert(it[3] == 4);
  }

  return true;
}

int main() {
  test();
  static_assert(test());

  return 0;
}
