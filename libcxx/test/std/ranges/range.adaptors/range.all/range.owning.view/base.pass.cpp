//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr R& base() & noexcept { return r_; }
// constexpr const R& base() const& noexcept { return r_; }
// constexpr R&& base() && noexcept { return std::move(r_); }
// constexpr const R&& base() const&& noexcept { return std::move(r_); }

#include <ranges>

#include <cassert>
#include <concepts>

#include "test_macros.h"

struct Base {
  int *begin() const;
  int *end() const;
};

constexpr bool test()
{
  using OwningView = std::ranges::owning_view<Base>;
  OwningView ov;
  decltype(auto) b1 = static_cast<OwningView&>(ov).base();
  decltype(auto) b2 = static_cast<OwningView&&>(ov).base();
  decltype(auto) b3 = static_cast<const OwningView&>(ov).base();
  decltype(auto) b4 = static_cast<const OwningView&&>(ov).base();

  ASSERT_SAME_TYPE(decltype(b1), Base&);
  ASSERT_SAME_TYPE(decltype(b2), Base&&);
  ASSERT_SAME_TYPE(decltype(b3), const Base&);
  ASSERT_SAME_TYPE(decltype(b4), const Base&&);

  assert(&b1 == &b2);
  assert(&b1 == &b3);
  assert(&b1 == &b4);

  ASSERT_NOEXCEPT(static_cast<OwningView&>(ov).base());
  ASSERT_NOEXCEPT(static_cast<OwningView&&>(ov).base());
  ASSERT_NOEXCEPT(static_cast<const OwningView&>(ov).base());
  ASSERT_NOEXCEPT(static_cast<const OwningView&&>(ov).base());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
