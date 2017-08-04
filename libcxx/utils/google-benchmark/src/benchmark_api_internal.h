#ifndef BENCHMARK_API_INTERNAL_H
#define BENCHMARK_API_INTERNAL_H

#include "benchmark/benchmark_api.h"

#include <cmath>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

namespace benchmark {
namespace internal {

// Information kept per benchmark we may want to run
struct Benchmark::Instance {
  std::string name;
  Benchmark* benchmark;
  ReportMode report_mode;
  std::vector<int> arg;
  TimeUnit time_unit;
  int range_multiplier;
  bool use_real_time;
  bool use_manual_time;
  BigO complexity;
  BigOFunc* complexity_lambda;
  UserCounters counters;
  bool last_benchmark_instance;
  int repetitions;
  double min_time;
  size_t iterations;
  int threads;  // Number of concurrent threads to us
};

bool FindBenchmarksInternal(const std::string& re,
                            std::vector<Benchmark::Instance>* benchmarks,
                            std::ostream* Err);

namespace {

bool IsZero(double n) {
  return std::abs(n) < std::numeric_limits<double>::epsilon();
}

}  // end namespace
}  // end namespace internal
}  // end namespace benchmark

#endif  // BENCHMARK_API_INTERNAL_H
