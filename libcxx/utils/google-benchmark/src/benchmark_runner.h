// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BENCHMARK_RUNNER_H_
#define BENCHMARK_RUNNER_H_

#include <thread>
#include <vector>

#include "benchmark_api_internal.h"
#include "internal_macros.h"
#include "perf_counters.h"
#include "thread_manager.h"

DECLARE_double(benchmark_min_time);

DECLARE_int32(benchmark_repetitions);

DECLARE_bool(benchmark_report_aggregates_only);

DECLARE_bool(benchmark_display_aggregates_only);

DECLARE_string(benchmark_perf_counters);

namespace benchmark {

namespace internal {

extern MemoryManager* memory_manager;

struct RunResults {
  std::vector<BenchmarkReporter::Run> non_aggregates;
  std::vector<BenchmarkReporter::Run> aggregates_only;

  bool display_report_aggregates_only = false;
  bool file_report_aggregates_only = false;
};

class BenchmarkRunner {
 public:
  BenchmarkRunner(const benchmark::internal::BenchmarkInstance& b_,
                  BenchmarkReporter::PerFamilyRunReports* reports_for_family);

  int GetNumRepeats() const { return repeats; }

  bool HasRepeatsRemaining() const {
    return GetNumRepeats() != num_repetitions_done;
  }

  void DoOneRepetition();

  RunResults&& GetResults();

  BenchmarkReporter::PerFamilyRunReports* GetReportsForFamily() const {
    return reports_for_family;
  };

 private:
  RunResults run_results;

  const benchmark::internal::BenchmarkInstance& b;
  BenchmarkReporter::PerFamilyRunReports* reports_for_family;

  const double min_time;
  const int repeats;
  const bool has_explicit_iteration_count;

  int num_repetitions_done = 0;

  std::vector<std::thread> pool;

  IterationCount iters;  // preserved between repetitions!
  // So only the first repetition has to find/calculate it,
  // the other repetitions will just use that precomputed iteration count.

  PerfCountersMeasurement perf_counters_measurement;
  PerfCountersMeasurement* const perf_counters_measurement_ptr;

  struct IterationResults {
    internal::ThreadManager::Result results;
    IterationCount iters;
    double seconds;
  };
  IterationResults DoNIterations();

  IterationCount PredictNumItersNeeded(const IterationResults& i) const;

  bool ShouldReportIterationResults(const IterationResults& i) const;
};

}  // namespace internal

}  // end namespace benchmark

#endif  // BENCHMARK_RUNNER_H_
