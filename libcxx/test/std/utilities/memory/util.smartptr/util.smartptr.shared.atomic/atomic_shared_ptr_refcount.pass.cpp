//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

#include <atomic>
#include <cassert>
#include <memory>
#include <utility> // Armv7, Armv8 require for std::move

namespace {
struct Tracker {
  static int alive;
  Tracker() { ++alive; }
  ~Tracker() { --alive; }
};
int Tracker::alive = 0;

struct ReentrantDeleter {
  std::atomic<std::shared_ptr<int>>* watch;
  void operator()(int* p) const {
    if (watch) {
      auto current = watch->load();
      (void)current;
    }
    delete p;
  }
};
} // namespace

int main(int, char**) {
  // store releases the previous value.
  {
    std::atomic<std::shared_ptr<Tracker>> a;
    a.store(std::make_shared<Tracker>());
    assert(Tracker::alive == 1);
    a.store(std::make_shared<Tracker>());
    assert(Tracker::alive == 1);
    a.store(nullptr);
    assert(Tracker::alive == 0);
  }

  // exchange returns old ownership and preserves refcounts.
  {
    auto first = std::make_shared<Tracker>();
    std::atomic<std::shared_ptr<Tracker>> a(first);
    assert(first.use_count() == 2);
    auto second = std::make_shared<Tracker>();
    auto old    = a.exchange(second);
    assert(old.get() == first.get());
    assert(first.use_count() == 2);
    assert(second.use_count() == 2);
    assert(Tracker::alive == 2);
  }

  // compare_exchange success/failure paths preserve ownership.
  {
    auto p1 = std::make_shared<Tracker>();
    auto p2 = std::make_shared<Tracker>();
    std::atomic<std::shared_ptr<Tracker>> a(p1);
    {
      std::shared_ptr<Tracker> expected = p2;
      bool ok                           = a.compare_exchange_strong(expected, p1);
      assert(!ok);
      assert(expected.get() == p1.get());
    }
    {
      std::shared_ptr<Tracker> expected = p1;
      bool ok                           = a.compare_exchange_strong(expected, p2);
      assert(ok);
    }
    assert(p1.use_count() == 1);
    assert(p2.use_count() == 2);
  }
  assert(Tracker::alive == 0);

  // atomic destruction releases held shared ownership.
  {
    auto p = std::make_shared<Tracker>();
    {
      std::atomic<std::shared_ptr<Tracker>> a(p);
      assert(p.use_count() == 2);
    }
    assert(p.use_count() == 1);
  }
  assert(Tracker::alive == 0);

  // Deleter must run after unlock; reentrant load must succeed.
  {
    std::atomic<std::shared_ptr<int>> a;
    {
      ReentrantDeleter d{&a};
      std::shared_ptr<int> p(new int(7), d);
      a.store(std::move(p));
    }
    a.store(std::shared_ptr<int>{});
  }

  return 0;
}
