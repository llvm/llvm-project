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
  std::atomic<A> a{A{0, 0}};
  auto r0 = a.load(std::memory_order_relaxed);
  assert(r0.x == 0);
  assert(r0.y == 0);
}
