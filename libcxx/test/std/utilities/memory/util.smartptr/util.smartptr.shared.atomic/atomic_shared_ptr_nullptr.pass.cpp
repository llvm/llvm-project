//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-threads

// LWG 3661 and LWG 3893 coverage.

#include <atomic>
#include <cassert>
#include <memory>
#include <type_traits>

int main(int, char**) {
  static_assert(std::is_constructible_v<std::atomic<std::shared_ptr<int>>, std::nullptr_t>);
  static_assert(std::is_constructible_v<std::atomic<std::weak_ptr<int>>>);
  static_assert(std::is_assignable_v<std::atomic<std::shared_ptr<int>>&, std::nullptr_t>);
  static_assert(std::is_assignable_v<std::atomic<std::shared_ptr<int>>&, std::shared_ptr<int>>);

  {
    std::atomic<std::shared_ptr<int>> a(nullptr);
    auto loaded = a.load();
    assert(!loaded);
  }

  {
    auto p = std::make_shared<int>(7);
    std::atomic<std::shared_ptr<int>> a(p);
    a           = nullptr;
    auto loaded = a.load();
    assert(!loaded);
  }

  {
    auto p = std::make_shared<int>(13);
    std::atomic<std::shared_ptr<int>> a;
    a           = p;
    auto loaded = a.load();
    assert(loaded && *loaded == 13);
  }

  {
    std::atomic<std::shared_ptr<int>> a;
    std::shared_ptr<int> got = a;
    assert(!got);
  }

  return 0;
}
