//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++20
// XFAIL: !has-128-bit-atomics

#include <atomic>
#include <cassert>
#include <cstdint>

#include "test_macros.h"

struct A {
  int64_t x = 0;
  int64_t y = 0;
};

int main() {
  // test number
  {
    std::atomic<A> a{A{0, 0}};
    auto r0 = a.load(std::memory_order_relaxed);
    assert(r0.x == 0);
    assert(r0.y == 0);

    auto expected = a.load(std::memory_order_relaxed);
    while (true) {
      if (a.compare_exchange_weak(expected, A{1, 2}, std::memory_order_acq_rel, std::memory_order_relaxed))
        break;
    }
    auto r1 = a.load(std::memory_order_relaxed);
    assert(r1.x == 1);
    assert(r1.y == 2);

    auto r2 = a.exchange(A{0, 0}, std::memory_order_relaxed);
    assert(r2.x == 1);
    assert(r2.y == 2);

    auto r3 = a.load(std::memory_order_relaxed);
    assert(r3.x == 0);
    assert(r3.y == 0);
  }
}
