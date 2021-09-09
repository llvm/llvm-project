//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr V base() const& requires copy_constructible<V>;
// constexpr V base() &&;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"

struct ContiguousView : std::ranges::view_base {
  int *ptr_;
  constexpr ContiguousView(int* ptr) : ptr_(ptr) {}
  constexpr ContiguousView(ContiguousView&&) = default;
  constexpr ContiguousView& operator=(ContiguousView&&) = default;
  friend constexpr int* begin(ContiguousView& view) { return view.ptr_; }
  friend constexpr int* begin(ContiguousView const& view) { return view.ptr_; }
  friend constexpr sentinel_wrapper<int*> end(ContiguousView& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
  friend constexpr sentinel_wrapper<int*> end(ContiguousView const& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
};

struct CopyableView : std::ranges::view_base {
  int *ptr_;
  constexpr CopyableView(int* ptr) : ptr_(ptr) {}
  friend constexpr int* begin(CopyableView& view) { return view.ptr_; }
  friend constexpr int* begin(CopyableView const& view) { return view.ptr_; }
  friend constexpr sentinel_wrapper<int*> end(CopyableView& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
  friend constexpr sentinel_wrapper<int*> end(CopyableView const& view) {
    return sentinel_wrapper<int*>{view.ptr_ + 8};
  }
};

constexpr bool hasLValueQualifiedBase(auto&& view) {
    return requires { view.base(); };
}

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::ranges::common_view<CopyableView> common(CopyableView{buffer});
    assert(common.base().ptr_ == buffer);
    assert(std::move(common).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(common.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(std::move(common).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(common));
  }

  {
    std::ranges::common_view<ContiguousView> common(ContiguousView{buffer});
    assert(std::move(common).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(std::move(common).base()), ContiguousView);
    static_assert(!hasLValueQualifiedBase(common));
  }

  {
    const std::ranges::common_view<CopyableView> common(CopyableView{buffer});
    assert(common.base().ptr_ == buffer);
    assert(std::move(common).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(common.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(std::move(common).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(common));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
