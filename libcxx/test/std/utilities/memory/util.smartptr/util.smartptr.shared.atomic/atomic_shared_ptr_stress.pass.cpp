//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

#include <array>
#include <atomic>
#include <cassert>
#include <memory>
#include <thread>
#include <vector>

namespace {
constexpr int kCandidateCount = 4;
constexpr int kWriters        = 4;
constexpr int kReaders        = 4;
constexpr int kIterations     = 4000;

template <class Atomic, class Make>
void run(Atomic& a, Make make_value, const std::array<int, kCandidateCount>& expected) {
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;

  for (int w = 0; w < kWriters; ++w) {
    threads.emplace_back([&, w] {
      for (int i = 0; i < kIterations; ++i) {
        a.store(make_value(expected[(w + i) % kCandidateCount]));
      }
    });
  }

  for (int r = 0; r < kReaders; ++r) {
    threads.emplace_back([&, r] {
      while (!stop.load(std::memory_order_relaxed)) {
        auto val = a.load();
        std::shared_ptr<int> sp;
        if constexpr (requires { val.lock(); }) {
          sp = val.lock();
        } else {
          sp = val;
        }
        if (sp) {
          int observed = *sp;
          bool ok      = false;
          for (int candidate : expected) {
            if (candidate == observed) {
              ok = true;
              break;
            }
          }
          assert(ok);
        }
        (void)r;
      }
    });
  }

  for (int w = 0; w < kWriters; ++w) {
    threads[w].join();
  }
  stop.store(true, std::memory_order_relaxed);
  for (int r = 0; r < kReaders; ++r) {
    threads[kWriters + r].join();
  }
}
} // namespace

int main(int, char**) {
  const std::array<int, kCandidateCount> expected = {1, 2, 3, 4};
  std::array<std::shared_ptr<int>, kCandidateCount> pool;
  for (int i = 0; i < kCandidateCount; ++i) {
    pool[i] = std::make_shared<int>(expected[i]);
  }

  {
    std::atomic<std::shared_ptr<int>> a(pool[0]);
    auto make_shared_value = [&](int v) {
      for (int i = 0; i < kCandidateCount; ++i) {
        if (expected[i] == v)
          return pool[i];
      }
      return pool[0];
    };
    run(a, make_shared_value, expected);
  }

  {
    std::atomic<std::weak_ptr<int>> a;
    a.store(std::weak_ptr<int>(pool[0]));
    auto make_weak_value = [&](int v) -> std::weak_ptr<int> {
      for (int i = 0; i < kCandidateCount; ++i) {
        if (expected[i] == v)
          return std::weak_ptr<int>(pool[i]);
      }
      return std::weak_ptr<int>(pool[0]);
    };
    run(a, make_weak_value, expected);
  }

  return 0;
}
