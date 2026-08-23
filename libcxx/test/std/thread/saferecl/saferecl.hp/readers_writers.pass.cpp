//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

// <hazard_pointer>

// Concurrent readers protecting a published object while writers replace and retire it. Every reader
// checks an invariant of the payload; under ASan/TSan this test also acts as a use-after-free / data-race
// detector for the implementation. No timing assumptions.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

struct Payload : std::hazard_pointer_obj_base<Payload> {
  std::uint64_t a;
  std::uint64_t b; // invariant: b == ~a while the object is alive
  explicit Payload(std::uint64_t v) : a(v), b(~v) {}
  ~Payload() {
    a = 0xDEADBEEF;
    b = 0xDEADBEEF; // break the invariant on destruction
  }
};

#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
constexpr int kReads  = 20000;
constexpr int kWrites = 2000;
#else
constexpr int kReads  = 200000;
constexpr int kWrites = 20000;
#endif
constexpr int kReaders = 4;
constexpr int kWriters = 2;

int main(int, char**) {
  std::atomic<Payload*> current{new Payload(1)};
  std::atomic<bool> go{false};

  std::vector<std::thread> threads;
  for (int r = 0; r < kReaders; ++r) {
    threads.push_back(support::make_test_thread([&] {
      while (!go.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kReads; ++i) {
        std::hazard_pointer h = std::make_hazard_pointer();
        Payload* p            = h.protect(current);
        if (p != nullptr) {
          std::uint64_t a = p->a;
          std::uint64_t b = p->b;
          assert(b == ~a);
        }
      }
    }));
  }
  for (int w = 0; w < kWriters; ++w) {
    threads.push_back(support::make_test_thread([&, w] {
      while (!go.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kWrites; ++i) {
        Payload* fresh = new Payload(static_cast<std::uint64_t>(w) * kWrites + i + 2);
        Payload* old   = current.exchange(fresh, std::memory_order_acq_rel);
        if (old != nullptr)
          old->retire();
      }
    }));
  }
  go.store(true, std::memory_order_release);
  for (std::thread& t : threads)
    t.join();
  Payload* last = current.exchange(nullptr);
  if (last != nullptr)
    last->retire();
  return 0;
}
