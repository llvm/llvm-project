//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++20
// XFAIL: !has-128-bit-atomics

#include <atomic>
#include <cassert>
#include <cstdint>

#include "test_macros.h"

struct Node {
  int i = 0;
};

struct Entry {
  Node* x = nullptr;
  Node* y = nullptr;
};

struct A {
  int64_t x = 0;
  int64_t y = 0;
};

int main() {
  // test number
  {
    std::atomic<A> a{};
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

  // test ptr
  {
    Node n1{1}, n2{2};
    std::atomic<Entry> a{};

    auto r0 = a.load(std::memory_order_relaxed);
    assert(r0.x == nullptr);
    assert(r0.y == nullptr);

    auto expected = a.load(std::memory_order_relaxed);
    while (true) {
      if (a.compare_exchange_weak(expected, Entry{&n1, &n2}, std::memory_order_acq_rel, std::memory_order_relaxed))
        break;
    }

    auto r1 = a.load(std::memory_order_relaxed);
    assert(r1.x == &n1);
    assert(r1.y == &n2);

    auto r2 = a.exchange(Entry{nullptr, nullptr}, std::memory_order_relaxed);
    assert(r2.x == &n1);
    assert(r2.y == &n2);

    auto r3 = a.load(std::memory_order_relaxed);
    assert(r3.x == nullptr);
    assert(r3.y == nullptr);
  }
}
