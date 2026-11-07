//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// XFAIL: libcpp-has-no-experimental-rcu
// XFAIL: availability-rcu-missing
// XFAIL: !has-128-bit-atomics
// REQUIRES: std-at-least-c++26

// void rcu_synchronize(rcu_domain& dom = rcu_default_domain()) noexcept;

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <new>
#include <random>
#include <rcu>
#include <stop_token>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

struct alignas(std::hardware_destructive_interference_size) PaddedVersion {
  std::atomic<std::uint64_t> value{0};
};

int main(int, char**) {
  {
    constexpr size_t num_reader_threads = 4;
    constexpr size_t num_sync_threads   = 4;
    constexpr int max_reader_iteration  = 10000;
    std::array<std::atomic<int>, num_reader_threads> lock_indicies{};
    std::array<std::atomic<int>, num_reader_threads> unlock_indicies{};
    auto& dom = std::rcu_default_domain();

    std::vector<std::jthread> reader_threads;
    for (size_t i = 0; i < num_reader_threads; ++i) {
      reader_threads.push_back(support::make_test_jthread([&, i] {
        for (int count = 0; count < max_reader_iteration; ++count) {
          dom.lock();
          lock_indicies[i].store(count, std::memory_order_relaxed);
          std::this_thread::sleep_for(std::chrono::microseconds(10));

          unlock_indicies[i].store(count, std::memory_order_relaxed);
          dom.unlock();
        }
      }));
    }

    std::vector<std::jthread> sync_threads;
    for (size_t i = 0; i < num_sync_threads; ++i) {
      sync_threads.push_back(support::make_test_jthread([&](std::stop_token st) {
        while (!st.stop_requested()) {
          std::array<int, num_reader_threads> pre_sync_reader_count{};
          for (size_t j = 0; j < num_reader_threads; ++j) {
            pre_sync_reader_count[j] = lock_indicies[j].load(std::memory_order_relaxed);
          }

          std::rcu_synchronize();
          for (size_t j = 0; j < num_reader_threads; ++j) {
            assert(unlock_indicies[j].load(std::memory_order_relaxed) >= pre_sync_reader_count[j]);
          }
        }
      }));
    }

    reader_threads.clear();
    sync_threads.clear();
  }

  {
    // Stress test for: if rcu_synchronize does not strongly happen before the
    // lock() opening a region R, then the unlock() closing R strongly happens
    // before rcu_synchronize returns. All atomics in the test are relaxed, so
    // any ordering has to come from lock()/unlock()/rcu_synchronize().
    //
    // The writer replaces version `old` with `old + 1` (fetch_add on
    // `published`), calls rcu_synchronize and then marks `old` as reclaimed.
    // Each reader enters a region, reads version `r`, records it in `inside[i]`
    // and keeps reading `reclaimed` before clearing `inside[i]` and leaving.
    // Two invariants follow:
    //   - a reader never sees `reclaimed >= r` inside the region where it read r;
    //   - after rcu_synchronize, the writer sees each `inside[i]` as 0 or > old.
    //
    // Readers run tight loops so that lock() races with rcu_synchronize millions
    // of times. Before each lock(), they write to cache lines shared by all
    // readers; those writes miss in the cache and delay lock()'s store of the
    // reader state. If lock() lacks a seq_cst fence, the reads in the region can
    // complete before that store is visible, rcu_synchronize misses the reader,
    // and one of the asserts fires.
    constexpr std::size_t num_reader_threads    = 4;
    constexpr std::uint32_t max_reads_in_region = 1024;
    constexpr auto test_duration                = std::chrono::seconds(1);

    PaddedVersion published;
    PaddedVersion reclaimed;
    PaddedVersion inside[num_reader_threads];
    published.value.store(1, std::memory_order_relaxed);

    constexpr std::size_t num_contended_lines = 16;
    PaddedVersion contended_lines[num_contended_lines];

    auto& dom = std::rcu_default_domain();

    std::vector<std::jthread> readers;
    for (std::size_t i = 0; i < num_reader_threads; ++i) {
      readers.push_back(support::make_test_jthread([&, i](std::stop_token st) {
        std::minstd_rand rng(static_cast<std::minstd_rand::result_type>(i + 1));
        std::uniform_int_distribution<std::uint32_t> reads_dist(0, max_reads_in_region - 1);
        std::uint64_t iteration = 0;
        while (!st.stop_requested()) {
          const std::uint32_t reads = reads_dist(rng);

          ++iteration;
          for (std::size_t k = 0; k < num_contended_lines; ++k) {
            contended_lines[(i + k) % num_contended_lines].value.store(iteration, std::memory_order_relaxed);
          }

          dom.lock();
          const std::uint64_t r = published.value.load(std::memory_order_relaxed);
          inside[i].value.store(r, std::memory_order_relaxed);
          for (std::uint32_t k = 0; k <= reads; ++k) {
            const std::uint64_t x = reclaimed.value.load(std::memory_order_relaxed);
            assert(x < r && "rcu_synchronize returned while a reader that saw the old version was still in its region");
          }
          inside[i].value.store(0, std::memory_order_relaxed);
          dom.unlock();
        }
      }));
    }

    std::jthread writer = support::make_test_jthread([&](std::stop_token st) {
      while (!st.stop_requested()) {
        const std::uint64_t old = published.value.fetch_add(1, std::memory_order_relaxed);

        std::rcu_synchronize(dom);

        for (std::size_t i = 0; i < num_reader_threads; ++i) {
          const std::uint64_t v = inside[i].value.load(std::memory_order_relaxed);
          assert((v == 0 || v > old) &&
                 "a reader region that saw the old version was still visible after rcu_synchronize");
        }

        std::uint64_t current = reclaimed.value.load(std::memory_order_relaxed);
        while (current < old && !reclaimed.value.compare_exchange_weak(current, old, std::memory_order_relaxed)) {
        }
      }
    });

    std::this_thread::sleep_for(test_duration);

    writer.request_stop();
    writer.join();
    readers.clear();
  }

  {
    // Lost wake-up test. rcu_synchronize sets a "waiting" flag, scans the
    // readers and sleeps until a reader leaving its region notifies it. If the
    // flag store is not ordered before the scan (missing seq_cst fence after
    // the store in rcu_synchronize), the writer can see a reader still inside
    // while that reader's unlock() still sees the flag unset and does not
    // notify, so rcu_synchronize sleeps forever.
    //
    // Each round has exactly one reader in a region. The writer waits until
    // the reader is inside, then calls rcu_synchronize while the reader unlocks
    // after a short random delay and then stays idle until the round ends, so
    // no other unlock() can wake the writer up. Before rcu_synchronize, the
    // writer writes to cache lines that another thread keeps writing; those
    // writes miss in the cache and delay the visibility of the flag store,
    // which widens the window. The main thread fails the test if a round does
    // not finish within `hang_timeout`.
    constexpr std::uint32_t max_unlock_delay  = 200;
    constexpr std::size_t num_contended_lines = 16;
    constexpr auto test_duration              = std::chrono::seconds(1);
    constexpr auto hang_timeout               = std::chrono::seconds(120);

    PaddedVersion round;
    PaddedVersion entered;
    PaddedVersion go;
    PaddedVersion finished;
    PaddedVersion contended_lines[num_contended_lines];
    std::atomic<bool> writer_exited{false};

    auto& dom = std::rcu_default_domain();

    std::jthread reader = support::make_test_jthread([&](std::stop_token st) {
      std::minstd_rand rng(1);
      std::uniform_int_distribution<std::uint32_t> delay_dist(0, max_unlock_delay);
      std::uint64_t last = 0;
      while (true) {
        std::uint64_t r;
        while ((r = round.value.load(std::memory_order_relaxed)) == last) {
          if (st.stop_requested())
            return;
        }
        last = r;

        dom.lock();
        entered.value.store(r, std::memory_order_relaxed);
        while (go.value.load(std::memory_order_relaxed) != r) {
        }
        const std::uint32_t delay = delay_dist(rng);
        for (std::uint32_t k = 0; k < delay; ++k) {
          DoNotOptimize(go.value.load(std::memory_order_relaxed));
        }
        dom.unlock();
      }
    });

    std::jthread noise = support::make_test_jthread([&](std::stop_token st) {
      std::uint64_t i = 0;
      while (!st.stop_requested()) {
        ++i;
        for (std::size_t k = 0; k < num_contended_lines; ++k) {
          contended_lines[k].value.store(i, std::memory_order_relaxed);
        }
      }
    });

    std::jthread writer = support::make_test_jthread([&](std::stop_token st) {
      for (std::uint64_t r = 1; !st.stop_requested(); ++r) {
        round.value.store(r, std::memory_order_relaxed);
        while (entered.value.load(std::memory_order_relaxed) != r) {
        }
        for (std::size_t k = 0; k < num_contended_lines; ++k) {
          contended_lines[k].value.store(r, std::memory_order_relaxed);
        }
        go.value.store(r, std::memory_order_relaxed);

        std::rcu_synchronize(dom);

        finished.value.store(r, std::memory_order_relaxed);
      }
      writer_exited.store(true, std::memory_order_relaxed);
    });

    const auto start         = std::chrono::steady_clock::now();
    auto last_progress_time  = start;
    std::uint64_t last_round = 0;
    while (!writer_exited.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      const auto now = std::chrono::steady_clock::now();
      if (now - start >= test_duration) {
        writer.request_stop();
      }
      const std::uint64_t f = finished.value.load(std::memory_order_relaxed);
      if (f != last_round) {
        last_round         = f;
        last_progress_time = now;
      }
      assert(now - last_progress_time < hang_timeout && "rcu_synchronize did not return after the reader left");
    }

    writer.join();
    reader.request_stop();
    reader.join();
    noise.request_stop();
    noise.join();
  }

  return 0;
}
