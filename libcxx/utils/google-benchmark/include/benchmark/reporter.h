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
#ifndef BENCHMARK_REPORTER_H_
#define BENCHMARK_REPORTER_H_

#include <cassert>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "benchmark_api.h"  // For forward declaration of BenchmarkReporter

namespace benchmark {

// Interface for custom benchmark result printers.
// By default, benchmark reports are printed to stdout. However an application
// can control the destination of the reports by calling
// RunSpecifiedBenchmarks and passing it a custom reporter object.
// The reporter object must implement the following interface.
class BenchmarkReporter {
 public:
  struct Context {
    int num_cpus;
    double mhz_per_cpu;
    bool cpu_scaling_enabled;

    // The number of chars in the longest benchmark name.
    size_t name_field_width;
  };

  struct Run {
    Run()
        : error_occurred(false),
          iterations(1),
          time_unit(kNanosecond),
          real_accumulated_time(0),
          cpu_accumulated_time(0),
          bytes_per_second(0),
          items_per_second(0),
          max_heapbytes_used(0),
          complexity(oNone),
          complexity_lambda(),
          complexity_n(0),
          report_big_o(false),
          report_rms(false) {}

    std::string benchmark_name;
    std::string report_label;  // Empty if not set by benchmark.
    bool error_occurred;
    std::string error_message;

    int64_t iterations;
    TimeUnit time_unit;
    double real_accumulated_time;
    double cpu_accumulated_time;

    // Return a value representing the real time per iteration in the unit
    // specified by 'time_unit'.
    // NOTE: If 'iterations' is zero the returned value represents the
    // accumulated time.
    double GetAdjustedRealTime() const;

    // Return a value representing the cpu time per iteration in the unit
    // specified by 'time_unit'.
    // NOTE: If 'iterations' is zero the returned value represents the
    // accumulated time.
    double GetAdjustedCPUTime() const;

    // Zero if not set by benchmark.
    double bytes_per_second;
    double items_per_second;

    // This is set to 0.0 if memory tracing is not enabled.
    double max_heapbytes_used;

    // Keep track of arguments to compute asymptotic complexity
    BigO complexity;
    BigOFunc* complexity_lambda;
    int complexity_n;

    // Inform print function whether the current run is a complexity report
    bool report_big_o;
    bool report_rms;
  };

  // Construct a BenchmarkReporter with the output stream set to 'std::cout'
  // and the error stream set to 'std::cerr'
  BenchmarkReporter();

  // Called once for every suite of benchmarks run.
  // The parameter "context" contains information that the
  // reporter may wish to use when generating its report, for example the
  // platform under which the benchmarks are running. The benchmark run is
  // never started if this function returns false, allowing the reporter
  // to skip runs based on the context information.
  virtual bool ReportContext(const Context& context) = 0;

  // Called once for each group of benchmark runs, gives information about
  // cpu-time and heap memory usage during the benchmark run. If the group
  // of runs contained more than two entries then 'report' contains additional
  // elements representing the mean and standard deviation of those runs.
  // Additionally if this group of runs was the last in a family of benchmarks
  // 'reports' contains additional entries representing the asymptotic
  // complexity and RMS of that benchmark family.
  virtual void ReportRuns(const std::vector<Run>& report) = 0;

  // Called once and only once after ever group of benchmarks is run and
  // reported.
  virtual void Finalize() {}

  // REQUIRES: The object referenced by 'out' is valid for the lifetime
  // of the reporter.
  void SetOutputStream(std::ostream* out) {
    assert(out);
    output_stream_ = out;
  }

  // REQUIRES: The object referenced by 'err' is valid for the lifetime
  // of the reporter.
  void SetErrorStream(std::ostream* err) {
    assert(err);
    error_stream_ = err;
  }

  std::ostream& GetOutputStream() const { return *output_stream_; }

  std::ostream& GetErrorStream() const { return *error_stream_; }

  virtual ~BenchmarkReporter();

  // Write a human readable string to 'out' representing the specified
  // 'context'.
  // REQUIRES: 'out' is non-null.
  static void PrintBasicContext(std::ostream* out, Context const& context);

 private:
  std::ostream* output_stream_;
  std::ostream* error_stream_;
};

// Simple reporter that outputs benchmark data to the console. This is the
// default reporter used by RunSpecifiedBenchmarks().
class ConsoleReporter : public BenchmarkReporter {
 public:
  enum OutputOptions { OO_None, OO_Color };
  explicit ConsoleReporter(OutputOptions color_output = OO_Color)
      : name_field_width_(0), color_output_(color_output == OO_Color) {}

  virtual bool ReportContext(const Context& context);
  virtual void ReportRuns(const std::vector<Run>& reports);

 protected:
  virtual void PrintRunData(const Run& report);
  size_t name_field_width_;

 private:
  bool color_output_;
};

class JSONReporter : public BenchmarkReporter {
 public:
  JSONReporter() : first_report_(true) {}
  virtual bool ReportContext(const Context& context);
  virtual void ReportRuns(const std::vector<Run>& reports);
  virtual void Finalize();

 private:
  void PrintRunData(const Run& report);

  bool first_report_;
};

class CSVReporter : public BenchmarkReporter {
 public:
  virtual bool ReportContext(const Context& context);
  virtual void ReportRuns(const std::vector<Run>& reports);

 private:
  void PrintRunData(const Run& report);
};

inline const char* GetTimeUnitString(TimeUnit unit) {
  switch (unit) {
    case kMillisecond:
      return "ms";
    case kMicrosecond:
      return "us";
    case kNanosecond:
    default:
      return "ns";
  }
}

inline double GetTimeUnitMultiplier(TimeUnit unit) {
  switch (unit) {
    case kMillisecond:
      return 1e3;
    case kMicrosecond:
      return 1e6;
    case kNanosecond:
    default:
      return 1e9;
  }
}

}  // end namespace benchmark
#endif  // BENCHMARK_REPORTER_H_
