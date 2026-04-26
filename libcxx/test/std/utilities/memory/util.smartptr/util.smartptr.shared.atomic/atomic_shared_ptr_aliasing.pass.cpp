//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-threads

// [util.smartptr.atomic.shared] equivalence checks with aliasing shared_ptr.

#include <atomic>
#include <cassert>
#include <memory>

struct Pair {
  int a;
  int b;
};

int main(int, char**) {
  auto owner = std::make_shared<Pair>(Pair{10, 20});

  std::shared_ptr<int> view_a(owner, &owner->a);
  std::shared_ptr<int> view_b(owner, &owner->b);
  assert(view_a.use_count() == view_b.use_count());
  assert(view_a.get() != view_b.get());

  std::atomic<std::shared_ptr<int>> atom(view_a);

  // Same control block, different stored pointer -> CAS must fail.
  {
    std::shared_ptr<int> expected = view_b;
    auto fresh                    = std::make_shared<int>(99);
    bool ok                       = atom.compare_exchange_strong(expected, fresh);
    assert(!ok);
    assert(expected.get() == view_a.get());
  }

  // Same stored pointer with the matching control block -> CAS must succeed.
  {
    std::shared_ptr<int> expected = view_a;
    bool ok                       = atom.compare_exchange_strong(expected, view_b);
    assert(ok);
    auto got = atom.load();
    assert(got.get() == view_b.get());
  }

  // Both empty are equivalent.
  {
    std::atomic<std::shared_ptr<int>> empty_atom;
    std::shared_ptr<int> expected;
    bool ok = empty_atom.compare_exchange_strong(expected, view_a);
    assert(ok);
    auto got = empty_atom.load();
    assert(got.get() == view_a.get());
  }

  // Empty vs non-empty are not equivalent.
  {
    std::atomic<std::shared_ptr<int>> non_empty(view_a);
    std::shared_ptr<int> expected;
    bool ok = non_empty.compare_exchange_strong(expected, view_b);
    assert(!ok);
    assert(expected.get() == view_a.get());
  }

  return 0;
}
