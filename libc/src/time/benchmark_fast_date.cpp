//===-- Benchmark for update_from_seconds_fast ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/time_utils.h"
#include "src/time/time_constants.h"
#include <cstdio>
#include <ctime>
#include <chrono>

using namespace LIBC_NAMESPACE;

// Benchmark helper
class Timer {
  std::chrono::high_resolution_clock::time_point start_time;
public:
  void start() {
    start_time = std::chrono::high_resolution_clock::now();
  }
  
  double elapsed_ms() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    return duration.count() / 1000.0;
  }
};

// Test data generator
void generate_test_timestamps(time_t* timestamps, int count, const char* pattern) {
  if (strcmp(pattern, "sequential") == 0) {
    // Sequential timestamps starting from year 2000
    time_t base = 946684800; // 2000-01-01
    for (int i = 0; i < count; i++) {
      timestamps[i] = base + i * 86400; // One day apart
    }
  } else if (strcmp(pattern, "random") == 0) {
    // Pseudo-random timestamps across a wide range
    time_t base = 0;
    for (int i = 0; i < count; i++) {
      timestamps[i] = base + (i * 123456789LL) % (100LL * 365 * 86400);
    }
  } else if (strcmp(pattern, "mixed") == 0) {
    // Mix of past, present, and future
    time_t ranges[] = {-2208988800LL, 0, 946684800, 1700000000, 4102444800LL};
    for (int i = 0; i < count; i++) {
      timestamps[i] = ranges[i % 5] + (i * 1000);
    }
  }
}

void benchmark_implementation(const char* name, 
                               int (*func)(time_t, struct tm*),
                               time_t* timestamps, 
                               int count) {
  Timer timer;
  struct tm result;
  
  timer.start();
  for (int i = 0; i < count; i++) {
    func(timestamps[i], &result);
  }
  double elapsed = timer.elapsed_ms();
  
  printf("  %s: %.2f ms (%.2f ns/conversion, %.2f M/sec)\n",
         name, elapsed, 
         elapsed * 1000000.0 / count,
         count / (elapsed * 1000.0));
}

void run_benchmark(const char* pattern, int count) {
  printf("\nBenchmark: %s pattern (%d conversions)\n", pattern, count);
  printf("========================================\n");
  
  time_t* timestamps = new time_t[count];
  generate_test_timestamps(timestamps, count, pattern);
  
  // Warm up cache
  struct tm result;
  for (int i = 0; i < 100; i++) {
    time_utils::update_from_seconds(timestamps[i % count], &result);
  }
  
  // Benchmark old implementation
  benchmark_implementation("Old algorithm ", 
                          time_utils::update_from_seconds,
                          timestamps, count);
  
  // Benchmark fast implementation  
  benchmark_implementation("Fast algorithm", 
                          time_utils::update_from_seconds_fast,
                          timestamps, count);
  
  delete[] timestamps;
}

int main() {
  printf("========================================\n");
  printf("Phase 4: Performance Benchmarks\n");
  printf("========================================\n");
  
  // Different workload patterns
  run_benchmark("sequential", 1000000);
  run_benchmark("random", 1000000);
  run_benchmark("mixed", 1000000);
  
  // Larger workload
  run_benchmark("sequential", 10000000);
  
  printf("\n✓ Benchmark complete\n");
  return 0;
}
