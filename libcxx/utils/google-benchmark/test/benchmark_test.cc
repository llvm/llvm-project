#include "benchmark/benchmark.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <utility>

#if defined(__GNUC__)
# define BENCHMARK_NOINLINE __attribute__((noinline))
#else
# define BENCHMARK_NOINLINE
#endif

namespace {

int BENCHMARK_NOINLINE Factorial(uint32_t n) {
  return (n == 1) ? 1 : n * Factorial(n - 1);
}

double CalculatePi(int depth) {
  double pi = 0.0;
  for (int i = 0; i < depth; ++i) {
    double numerator = static_cast<double>(((i % 2) * 2) - 1);
    double denominator = static_cast<double>((2 * i) - 1);
    pi += numerator / denominator;
  }
  return (pi - 1.0) * 4;
}

std::set<int> ConstructRandomSet(int size) {
  std::set<int> s;
  for (int i = 0; i < size; ++i)
    s.insert(i);
  return s;
}

std::mutex test_vector_mu;
std::vector<int>* test_vector = nullptr;

}  // end namespace

static void BM_Factorial(benchmark::State& state) {
  int fac_42 = 0;
  while (state.KeepRunning())
    fac_42 = Factorial(8);
  // Prevent compiler optimizations
  std::stringstream ss;
  ss << fac_42;
  state.SetLabel(ss.str());
}
BENCHMARK(BM_Factorial);
BENCHMARK(BM_Factorial)->UseRealTime();

static void BM_CalculatePiRange(benchmark::State& state) {
  double pi = 0.0;
  while (state.KeepRunning())
    pi = CalculatePi(state.range_x());
  std::stringstream ss;
  ss << pi;
  state.SetLabel(ss.str());
}
BENCHMARK_RANGE(BM_CalculatePiRange, 1, 1024 * 1024);

static void BM_CalculatePi(benchmark::State& state) {
  static const int depth = 1024;
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(CalculatePi(depth));
  }
}
BENCHMARK(BM_CalculatePi)->Threads(8);
BENCHMARK(BM_CalculatePi)->ThreadRange(1, 32);
BENCHMARK(BM_CalculatePi)->ThreadPerCpu();

static void BM_SetInsert(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    std::set<int> data = ConstructRandomSet(state.range_x());
    state.ResumeTiming();
    for (int j = 0; j < state.range_y(); ++j)
      data.insert(rand());
  }
  state.SetItemsProcessed(state.iterations() * state.range_y());
  state.SetBytesProcessed(state.iterations() * state.range_y() * sizeof(int));
}
BENCHMARK(BM_SetInsert)->RangePair(1<<10,8<<10, 1,10);

template<typename Container, typename ValueType = typename Container::value_type>
static void BM_Sequential(benchmark::State& state) {
  ValueType v = 42;
  while (state.KeepRunning()) {
    Container c;
    for (int i = state.range_x(); --i; )
      c.push_back(v);
  }
  const size_t items_processed = state.iterations() * state.range_x();
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(v));
}
BENCHMARK_TEMPLATE2(BM_Sequential, std::vector<int>, int)->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Sequential, std::list<int>)->Range(1 << 0, 1 << 10);
// Test the variadic version of BENCHMARK_TEMPLATE in C++11 and beyond.
#if __cplusplus >= 201103L
BENCHMARK_TEMPLATE(BM_Sequential, std::vector<int>, int)->Arg(512);
#endif

static void BM_StringCompare(benchmark::State& state) {
  std::string s1(state.range_x(), '-');
  std::string s2(state.range_x(), '-');
  while (state.KeepRunning())
    benchmark::DoNotOptimize(s1.compare(s2));
}
BENCHMARK(BM_StringCompare)->Range(1, 1<<20);

static void BM_SetupTeardown(benchmark::State& state) {
  if (state.thread_index == 0) {
    // No need to lock test_vector_mu here as this is running single-threaded.
    test_vector = new std::vector<int>();
  }
  int i = 0;
  while (state.KeepRunning()) {
    std::lock_guard<std::mutex> l(test_vector_mu);
    if (i%2 == 0)
      test_vector->push_back(i);
    else
      test_vector->pop_back();
    ++i;
  }
  if (state.thread_index == 0) {
    delete test_vector;
  }
}
BENCHMARK(BM_SetupTeardown)->ThreadPerCpu();

static void BM_LongTest(benchmark::State& state) {
  double tracker = 0.0;
  while (state.KeepRunning()) {
    for (int i = 0; i < state.range_x(); ++i)
      benchmark::DoNotOptimize(tracker += i);
  }
}
BENCHMARK(BM_LongTest)->Range(1<<16,1<<28);

static void BM_ParallelMemset(benchmark::State& state) {
  int size = state.range_x() / sizeof(int);
  int thread_size = size / state.threads;
  int from = thread_size * state.thread_index;
  int to = from + thread_size;

  if (state.thread_index == 0) {
    test_vector = new std::vector<int>(size);
  }

  while (state.KeepRunning()) {
    for (int i = from; i < to; i++) {
      // No need to lock test_vector_mu as ranges
      // do not overlap between threads.
      benchmark::DoNotOptimize(test_vector->at(i) = 1);
    }
  }

  if (state.thread_index == 0) {
    delete test_vector;
  }
}
BENCHMARK(BM_ParallelMemset)->Arg(10 << 20)->ThreadRange(1, 4);

static void BM_ManualTiming(benchmark::State& state) {
  size_t slept_for = 0;
  int microseconds = state.range_x();
  std::chrono::duration<double, std::micro> sleep_duration {
    static_cast<double>(microseconds)
  };

  while (state.KeepRunning()) {
    auto start   = std::chrono::high_resolution_clock::now();
    // Simulate some useful workload with a sleep
    std::this_thread::sleep_for(std::chrono::duration_cast<
      std::chrono::nanoseconds>(sleep_duration));
    auto end     = std::chrono::high_resolution_clock::now();

    auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    state.SetIterationTime(elapsed.count());
    slept_for += microseconds;
  }
  state.SetItemsProcessed(slept_for);
}
BENCHMARK(BM_ManualTiming)->Range(1, 1 << 14)->UseRealTime();
BENCHMARK(BM_ManualTiming)->Range(1, 1 << 14)->UseManualTime();

#if __cplusplus >= 201103L

template <class ...Args>
void BM_with_args(benchmark::State& state, Args&&...) {
  while (state.KeepRunning()) {}
}
BENCHMARK_CAPTURE(BM_with_args, int_test, 42, 43, 44);
BENCHMARK_CAPTURE(BM_with_args, string_and_pair_test,
                  std::string("abc"), std::pair<int, double>(42, 3.8));

void BM_non_template_args(benchmark::State& state, int, double) {
  while(state.KeepRunning()) {}
}
BENCHMARK_CAPTURE(BM_non_template_args, basic_test, 0, 0);

#endif // __cplusplus >= 201103L

BENCHMARK_MAIN()

