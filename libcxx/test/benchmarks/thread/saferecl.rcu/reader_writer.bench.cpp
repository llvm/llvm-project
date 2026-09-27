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
// REQUIRES: std-at-least-c++26

// <rcu>

#include <atomic>
#include <benchmark/benchmark.h>
#include <chrono>
#include <rcu>
#include <string>
#include <thread>
#include <vector>

#include "test_macros.h"

namespace {

constexpr auto test_time = std::chrono::seconds(10);

struct alignas(128) MyObject : public std::rcu_obj_base<MyObject> {
  std::string data_;

  MyObject() : data_("instance very very very long string") {}

  void doWork() noexcept { benchmark::DoNotOptimize(data_); }
};

struct ThreadStats {
  uint64_t operations = 0;
};

struct BenchmarkContext {
  std::rcu_domain& domain = std::rcu_default_domain();

  std::atomic<MyObject*> global_object = new MyObject();

  std::atomic<bool> stop{false};

  std::vector<ThreadStats> reader_stats;
  std::vector<ThreadStats> writer_stats;

  BenchmarkContext(int nr, int nw) : reader_stats(nr), writer_stats(nw) {}

  ~BenchmarkContext() {
    // All workers have stopped and therefore no new
    // objects can be retired at this point.
    std::rcu_barrier(domain);

    MyObject* object = global_object.load(std::memory_order_relaxed);

    delete object;
  }

  BenchmarkContext(const BenchmarkContext&)            = delete;
  BenchmarkContext& operator=(const BenchmarkContext&) = delete;
};

void reader_thread(BenchmarkContext& context, int index) {
  uint64_t count = 0;

  while (!context.stop.load(std::memory_order_relaxed)) {
    context.domain.lock();

    MyObject* object = context.global_object.load(std::memory_order_relaxed);

    benchmark::DoNotOptimize(object);

    object->doWork();

    context.domain.unlock();

    ++count;
  }

  context.reader_stats[index].operations = count;
}

void writer_thread(BenchmarkContext& context, int index) {
  uint64_t count = 0;

  while (!context.stop.load(std::memory_order_relaxed)) {
    MyObject* new_object = new MyObject();

    MyObject* old_object = context.global_object.exchange(new_object, std::memory_order_acq_rel);

    old_object->retire();

    ++count;
  }

  context.writer_stats[index].operations = count;
}

void syncer_thread(BenchmarkContext& context) {
  while (!context.stop.load(std::memory_order_relaxed)) {
    std::rcu_barrier(context.domain);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

static void BM_RCU(benchmark::State& state) {
  const int num_reader = state.range(0);
  const int num_writer = state.range(1);
  for (auto _ : state) {
    BenchmarkContext context(num_writer, num_writer);

    std::vector<std::jthread> readers;
    std::vector<std::jthread> writers;

    readers.reserve(num_reader);
    writers.reserve(num_writer);

    for (int i = 0; i < num_reader; ++i) {
      readers.emplace_back([&context, i] { reader_thread(context, i); });
    }

    for (int i = 0; i < num_writer; ++i) {
      writers.emplace_back([&context, i] { writer_thread(context, i); });
    }

    std::jthread syncer([&context] { syncer_thread(context); });

    std::this_thread::yield();

    auto start = std::chrono::steady_clock::now();

    std::this_thread::sleep_for(test_time);

    auto end = std::chrono::steady_clock::now();

    context.stop.store(true, std::memory_order_relaxed);

    syncer.join();

    for (auto& reader : readers)
      reader.join();

    for (auto& writer : writers)
      writer.join();

    std::rcu_barrier(context.domain);

    const double seconds = std::chrono::duration<double>(end - start).count();

    uint64_t total_reads  = 0;
    uint64_t total_writes = 0;

    for (const auto& stats : context.reader_stats)
      total_reads += stats.operations;

    for (const auto& stats : context.writer_stats)
      total_writes += stats.operations;

    state.counters["read/s"] = benchmark::Counter(static_cast<double>(total_reads) / seconds);

    state.counters["write/s"] = benchmark::Counter(static_cast<double>(total_writes) / seconds);

    for (int i = 0; i < num_reader; ++i) {
      state.counters["reader_" + std::to_string(i) + "/s"] =
          benchmark::Counter(static_cast<double>(context.reader_stats[i].operations) / seconds);
    }

    for (int i = 0; i < num_writer; ++i) {
      state.counters["writer_" + std::to_string(i) + "/s"] =
          benchmark::Counter(static_cast<double>(context.writer_stats[i].operations) / seconds);
    }

    state.counters["total/s"] = benchmark::Counter(static_cast<double>(total_reads + total_writes) / seconds);
  }
}

} // namespace

BENCHMARK(BM_RCU)
    ->Args({4, 4})
    ->Args({4, 1})
    ->ArgNames({"readers", "writers"})
    ->Iterations(1)
    ->UseRealTime();

BENCHMARK_MAIN();
