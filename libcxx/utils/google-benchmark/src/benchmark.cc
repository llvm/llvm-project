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

#include "benchmark/benchmark.h"
#include "internal_macros.h"

#ifndef BENCHMARK_OS_WINDOWS
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <fstream>
#include <memory>
#include <thread>

#include "check.h"
#include "commandlineflags.h"
#include "complexity.h"
#include "log.h"
#include "mutex.h"
#include "re.h"
#include "stat.h"
#include "string_util.h"
#include "sysinfo.h"
#include "walltime.h"

DEFINE_bool(benchmark_list_tests, false,
            "Print a list of benchmarks. This option overrides all other "
            "options.");

DEFINE_string(benchmark_filter, ".",
              "A regular expression that specifies the set of benchmarks "
              "to execute.  If this flag is empty, no benchmarks are run.  "
              "If this flag is the string \"all\", all benchmarks linked "
              "into the process are run.");

DEFINE_double(benchmark_min_time, 0.5,
              "Minimum number of seconds we should run benchmark before "
              "results are considered significant.  For cpu-time based "
              "tests, this is the lower bound on the total cpu time "
              "used by all threads that make up the test.  For real-time "
              "based tests, this is the lower bound on the elapsed time "
              "of the benchmark execution, regardless of number of "
              "threads.");

DEFINE_int32(benchmark_repetitions, 1,
             "The number of runs of each benchmark. If greater than 1, the "
             "mean and standard deviation of the runs will be reported.");

DEFINE_string(benchmark_format, "console",
              "The format to use for console output. Valid values are "
              "'console', 'json', or 'csv'.");

DEFINE_string(benchmark_out_format, "json",
              "The format to use for file output. Valid values are "
              "'console', 'json', or 'csv'.");

DEFINE_string(benchmark_out, "", "The file to write additonal output to");

DEFINE_bool(color_print, true, "Enables colorized logging.");

DEFINE_int32(v, 0, "The level of verbose logging to output");


namespace benchmark {

namespace internal {

void UseCharPointer(char const volatile*) {}

// NOTE: This is a dummy "mutex" type used to denote the actual mutex
// returned by GetBenchmarkLock(). This is only used to placate the thread
// safety warnings by giving the return of GetBenchmarkLock() a name.
struct CAPABILITY("mutex") BenchmarkLockType {};
BenchmarkLockType BenchmarkLockVar;

} // end namespace internal

inline Mutex& RETURN_CAPABILITY(::benchmark::internal::BenchmarkLockVar)
GetBenchmarkLock()
{
  static Mutex lock;
  return lock;
}

namespace {

bool IsZero(double n) {
    return std::abs(n) < std::numeric_limits<double>::epsilon();
}

// For non-dense Range, intermediate values are powers of kRangeMultiplier.
static const int kRangeMultiplier = 8;
static const size_t kMaxIterations = 1000000000;

bool running_benchmark = false;

// Global variable so that a benchmark can cause a little extra printing
std::string* GetReportLabel() {
    static std::string label GUARDED_BY(GetBenchmarkLock());
    return &label;
}

// Global variable so that a benchmark can report an error as a human readable
// string. If error_message is null no error occurred.
#if defined(_MSC_VER) && _MSC_VER <= 1800
typedef char* error_message_type;
#else
typedef const char* error_message_type;
#endif

static std::atomic<error_message_type> error_message = ATOMIC_VAR_INIT(nullptr);

// TODO(ericwf): support MallocCounter.
//static benchmark::MallocCounter *benchmark_mc;

struct ThreadStats {
    ThreadStats() : bytes_processed(0), items_processed(0), complexity_n(0) {}
    int64_t bytes_processed;
    int64_t items_processed;
    int  complexity_n;
};

// Timer management class
class TimerManager {
 public:
  TimerManager(int num_threads, Notification* done)
      : num_threads_(num_threads),
        running_threads_(num_threads),
        done_(done),
        running_(false),
        real_time_used_(0),
        cpu_time_used_(0),
        manual_time_used_(0),
        num_finalized_(0),
        phase_number_(0),
        entered_(0)
  {
  }

  // Called by each thread
  void StartTimer() EXCLUDES(lock_) {
    bool last_thread = false;
    {
      MutexLock ml(lock_);
      last_thread = Barrier(ml);
      if (last_thread) {
        CHECK(!running_) << "Called StartTimer when timer is already running";
        running_ = true;
        start_real_time_ = walltime::Now();
        start_cpu_time_ = MyCPUUsage() + ChildrenCPUUsage();
       }
     }
     if (last_thread) {
       phase_condition_.notify_all();
     }
  }

  // Called by each thread
  void StopTimer() EXCLUDES(lock_) {
    bool last_thread = false;
    {
      MutexLock ml(lock_);
      last_thread = Barrier(ml);
      if (last_thread) {
        CHECK(running_) << "Called StopTimer when timer is already stopped";
        InternalStop();
      }
    }
    if (last_thread) {
      phase_condition_.notify_all();
    }
  }

  // Called by each thread
  void SetIterationTime(double seconds) EXCLUDES(lock_) {
    bool last_thread = false;
    {
      MutexLock ml(lock_);
      last_thread = Barrier(ml);
      if (last_thread) {
        manual_time_used_ += seconds;
      }
    }
    if (last_thread) {
      phase_condition_.notify_all();
    }
  }

  // Called by each thread
  void Finalize() EXCLUDES(lock_) {
    MutexLock l(lock_);
    num_finalized_++;
    if (num_finalized_ == num_threads_) {
      CHECK(!running_) <<
        "The timer should be stopped before the timer is finalized";
      done_->Notify();
    }
  }

  void RemoveErroredThread() EXCLUDES(lock_) {
    MutexLock ml(lock_);
    int last_thread = --running_threads_ == 0;
    if (last_thread && running_)
      InternalStop();
    else if (!last_thread)
      phase_condition_.notify_all();
  }

  // REQUIRES: timer is not running
  double real_time_used() EXCLUDES(lock_) {
    MutexLock l(lock_);
    CHECK(!running_);
    return real_time_used_;
  }

  // REQUIRES: timer is not running
  double cpu_time_used() EXCLUDES(lock_) {
    MutexLock l(lock_);
    CHECK(!running_);
    return cpu_time_used_;
  }

  // REQUIRES: timer is not running
  double manual_time_used() EXCLUDES(lock_) {
    MutexLock l(lock_);
    CHECK(!running_);
    return manual_time_used_;
  }

 private:
  Mutex lock_;
  Condition phase_condition_;
  int num_threads_;
  int running_threads_;
  Notification* done_;

  bool running_;                // Is the timer running
  double start_real_time_;      // If running_
  double start_cpu_time_;       // If running_

  // Accumulated time so far (does not contain current slice if running_)
  double real_time_used_;
  double cpu_time_used_;
  // Manually set iteration time. User sets this with SetIterationTime(seconds).
  double manual_time_used_;

  // How many threads have called Finalize()
  int num_finalized_;

  // State for barrier management
  int phase_number_;
  int entered_;         // Number of threads that have entered this barrier

  void InternalStop() REQUIRES(lock_) {
    CHECK(running_);
    running_ = false;
    real_time_used_ += walltime::Now() - start_real_time_;
    cpu_time_used_ += ((MyCPUUsage() + ChildrenCPUUsage())
                       - start_cpu_time_);
  }

  // Enter the barrier and wait until all other threads have also
  // entered the barrier.  Returns iff this is the last thread to
  // enter the barrier.
  bool Barrier(MutexLock& ml) REQUIRES(lock_) {
    CHECK_LT(entered_, running_threads_);
    entered_++;
    if (entered_ < running_threads_) {
      // Wait for all threads to enter
      int phase_number_cp = phase_number_;
      auto cb = [this, phase_number_cp]() {
        return this->phase_number_ > phase_number_cp ||
               entered_ == running_threads_; // A thread has aborted in error
      };
      phase_condition_.wait(ml.native_handle(), cb);
      if (phase_number_ > phase_number_cp)
        return false;
      // else (running_threads_ == entered_) and we are the last thread.
    }
    // Last thread has reached the barrier
    phase_number_++;
    entered_ = 0;
    return true;
  }
};

// TimerManager for current run.
static std::unique_ptr<TimerManager> timer_manager = nullptr;

} // end namespace

namespace internal {

// Information kept per benchmark we may want to run
struct Benchmark::Instance {
  std::string      name;
  Benchmark*       benchmark;
  std::vector<int> arg;
  TimeUnit         time_unit;
  int              range_multiplier;
  bool             use_real_time;
  bool             use_manual_time;
  BigO             complexity;
  BigOFunc*        complexity_lambda;
  bool             last_benchmark_instance;
  int              repetitions;
  double           min_time;
  int              threads;    // Number of concurrent threads to use
  bool             multithreaded;  // Is benchmark multi-threaded?
};

// Class for managing registered benchmarks.  Note that each registered
// benchmark identifies a family of related benchmarks to run.
class BenchmarkFamilies {
 public:
  static BenchmarkFamilies* GetInstance();

  // Registers a benchmark family and returns the index assigned to it.
  size_t AddBenchmark(std::unique_ptr<Benchmark> family);

  // Extract the list of benchmark instances that match the specified
  // regular expression.
  bool FindBenchmarks(const std::string& re,
                      std::vector<Benchmark::Instance>* benchmarks);
 private:
  BenchmarkFamilies() {}

  std::vector<std::unique_ptr<Benchmark>> families_;
  Mutex mutex_;
};


class BenchmarkImp {
public:
  explicit BenchmarkImp(const char* name);
  ~BenchmarkImp();

  void Arg(int x);
  void Unit(TimeUnit unit);
  void Range(int start, int limit);
  void DenseRange(int start, int limit, int step = 1);
  void Args(const std::vector<int>& args);
  void Ranges(const std::vector<std::pair<int, int>>& ranges);
  void RangeMultiplier(int multiplier);
  void MinTime(double n);
  void Repetitions(int n);
  void UseRealTime();
  void UseManualTime();
  void Complexity(BigO complexity);
  void ComplexityLambda(BigOFunc* complexity);
  void Threads(int t);
  void ThreadRange(int min_threads, int max_threads);
  void ThreadPerCpu();
  void SetName(const char* name);

  static void AddRange(std::vector<int>* dst, int lo, int hi, int mult);

  int ArgsCnt() const { return args_.empty() ? -1 : static_cast<int>(args_.front().size()); }

private:
  friend class BenchmarkFamilies;

  std::string name_;
  std::vector< std::vector<int> > args_;  // Args for all benchmark runs
  TimeUnit time_unit_;
  int range_multiplier_;
  double min_time_;
  int repetitions_;
  bool use_real_time_;
  bool use_manual_time_;
  BigO complexity_;
  BigOFunc* complexity_lambda_;
  std::vector<int> thread_counts_;

  BenchmarkImp& operator=(BenchmarkImp const&);
};

BenchmarkFamilies* BenchmarkFamilies::GetInstance() {
  static BenchmarkFamilies instance;
  return &instance;
}


size_t BenchmarkFamilies::AddBenchmark(std::unique_ptr<Benchmark> family) {
  MutexLock l(mutex_);
  size_t index = families_.size();
  families_.push_back(std::move(family));
  return index;
}

bool BenchmarkFamilies::FindBenchmarks(
    const std::string& spec,
    std::vector<Benchmark::Instance>* benchmarks) {
  // Make regular expression out of command-line flag
  std::string error_msg;
  Regex re;
  if (!re.Init(spec, &error_msg)) {
    std::cerr << "Could not compile benchmark re: " << error_msg << std::endl;
    return false;
  }

  // Special list of thread counts to use when none are specified
  std::vector<int> one_thread;
  one_thread.push_back(1);

  MutexLock l(mutex_);
  for (std::unique_ptr<Benchmark>& bench_family : families_) {
    // Family was deleted or benchmark doesn't match
    if (!bench_family) continue;
    BenchmarkImp* family = bench_family->imp_;

    if (family->ArgsCnt() == -1) {
      family->Args({});
    }

    for (auto const& args : family->args_) {
      const std::vector<int>* thread_counts =
        (family->thread_counts_.empty()
         ? &one_thread
         : &family->thread_counts_);
      for (int num_threads : *thread_counts) {

        Benchmark::Instance instance;
        instance.name = family->name_;
        instance.benchmark = bench_family.get();
        instance.arg = args;
        instance.time_unit = family->time_unit_;
        instance.range_multiplier = family->range_multiplier_;
        instance.min_time = family->min_time_;
        instance.repetitions = family->repetitions_;
        instance.use_real_time = family->use_real_time_;
        instance.use_manual_time = family->use_manual_time_;
        instance.complexity = family->complexity_;
        instance.complexity_lambda = family->complexity_lambda_;
        instance.threads = num_threads;
        instance.multithreaded = !(family->thread_counts_.empty());

        // Add arguments to instance name
        for (auto const& arg : args) {
          AppendHumanReadable(arg, &instance.name);
        }

        if (!IsZero(family->min_time_)) {
          instance.name +=  StringPrintF("/min_time:%0.3f",  family->min_time_);
        }
        if (family->repetitions_ != 0) {
          instance.name +=  StringPrintF("/repeats:%d",  family->repetitions_);
        }
        if (family->use_manual_time_) {
          instance.name +=  "/manual_time";
        } else if (family->use_real_time_) {
          instance.name +=  "/real_time";
        }

        // Add the number of threads used to the name
        if (!family->thread_counts_.empty()) {
          instance.name += StringPrintF("/threads:%d", instance.threads);
        }

        if (re.Match(instance.name)) {
          instance.last_benchmark_instance = (args == family->args_.back());
          benchmarks->push_back(instance);
        }
      }
    }
  }
  return true;
}

BenchmarkImp::BenchmarkImp(const char* name)
    : name_(name), time_unit_(kNanosecond),
      range_multiplier_(kRangeMultiplier), min_time_(0.0), repetitions_(0),
      use_real_time_(false), use_manual_time_(false),
      complexity_(oNone) {
}

BenchmarkImp::~BenchmarkImp() {
}

void BenchmarkImp::Arg(int x) {
  CHECK(ArgsCnt() == -1 || ArgsCnt() == 1);
  args_.push_back({x});
}

void BenchmarkImp::Unit(TimeUnit unit) {
  time_unit_ = unit;
}

void BenchmarkImp::Range(int start, int limit) {
  CHECK(ArgsCnt() == -1 || ArgsCnt() == 1);
  std::vector<int> arglist;
  AddRange(&arglist, start, limit, range_multiplier_);

  for (int i : arglist) {
    args_.push_back({i});
  }
}

void BenchmarkImp::DenseRange(int start, int limit, int step) {
  CHECK(ArgsCnt() == -1 || ArgsCnt() == 1);
  CHECK_GE(start, 0);
  CHECK_LE(start, limit);
  for (int arg = start; arg <= limit; arg+= step) {
    args_.push_back({arg});
  }
}

void BenchmarkImp::Args(const std::vector<int>& args)
{
  args_.push_back(args);
}

void BenchmarkImp::Ranges(const std::vector<std::pair<int, int>>& ranges) {
  std::vector<std::vector<int>> arglists(ranges.size());
  int total = 1;
  for (std::size_t i = 0; i < ranges.size(); i++) {
    AddRange(&arglists[i], ranges[i].first, ranges[i].second, range_multiplier_);
    total *= arglists[i].size();
  }

  std::vector<std::size_t> ctr(total, 0);

  for (int i = 0; i < total; i++) {
    std::vector<int> tmp;

    for (std::size_t j = 0; j < arglists.size(); j++) {
      tmp.push_back(arglists[j][ctr[j]]);
    }

    args_.push_back(tmp);

    for (std::size_t j = 0; j < arglists.size(); j++) {
      if (ctr[j] + 1 < arglists[j].size()) {
        ++ctr[j];
        break;
      }
      ctr[j] = 0;
    }
  }
}

void BenchmarkImp::RangeMultiplier(int multiplier) {
  CHECK(multiplier > 1);
  range_multiplier_ = multiplier;
}

void BenchmarkImp::MinTime(double t) {
  CHECK(t > 0.0);
  min_time_ = t;
}


void BenchmarkImp::Repetitions(int n) {
  CHECK(n > 0);
  repetitions_ = n;
}

void BenchmarkImp::UseRealTime() {
  CHECK(!use_manual_time_) << "Cannot set UseRealTime and UseManualTime simultaneously.";
  use_real_time_ = true;
}

void BenchmarkImp::UseManualTime() {
  CHECK(!use_real_time_) << "Cannot set UseRealTime and UseManualTime simultaneously.";
  use_manual_time_ = true;
}

void BenchmarkImp::Complexity(BigO complexity){
  complexity_ = complexity;
}

void BenchmarkImp::ComplexityLambda(BigOFunc* complexity) {
  complexity_lambda_ = complexity;
}

void BenchmarkImp::Threads(int t) {
  CHECK_GT(t, 0);
  thread_counts_.push_back(t);
}

void BenchmarkImp::ThreadRange(int min_threads, int max_threads) {
  CHECK_GT(min_threads, 0);
  CHECK_GE(max_threads, min_threads);

  AddRange(&thread_counts_, min_threads, max_threads, 2);
}

void BenchmarkImp::ThreadPerCpu() {
  static int num_cpus = NumCPUs();
  thread_counts_.push_back(num_cpus);
}

void BenchmarkImp::SetName(const char* name) {
  name_ = name;
}

void BenchmarkImp::AddRange(std::vector<int>* dst, int lo, int hi, int mult) {
  CHECK_GE(lo, 0);
  CHECK_GE(hi, lo);
  CHECK_GE(mult, 2);

  // Add "lo"
  dst->push_back(lo);

  static const int kint32max = std::numeric_limits<int32_t>::max();

  // Now space out the benchmarks in multiples of "mult"
  for (int32_t i = 1; i < kint32max/mult; i *= mult) {
    if (i >= hi) break;
    if (i > lo) {
      dst->push_back(i);
    }
  }
  // Add "hi" (if different from "lo")
  if (hi != lo) {
    dst->push_back(hi);
  }
}

Benchmark::Benchmark(const char* name)
    : imp_(new BenchmarkImp(name))
{
}

Benchmark::~Benchmark()  {
  delete imp_;
}

Benchmark::Benchmark(Benchmark const& other)
  : imp_(new BenchmarkImp(*other.imp_))
{
}

Benchmark* Benchmark::Arg(int x) {
  CHECK(imp_->ArgsCnt() == -1 || imp_->ArgsCnt() == 1);
  imp_->Arg(x);
  return this;
}

Benchmark* Benchmark::Unit(TimeUnit unit) {
  imp_->Unit(unit);
  return this;
}

Benchmark* Benchmark::Range(int start, int limit) {
  CHECK(imp_->ArgsCnt() == -1 || imp_->ArgsCnt() == 1);
  imp_->Range(start, limit);
  return this;
}

Benchmark* Benchmark::Ranges(const std::vector<std::pair<int, int>>& ranges)
{
  CHECK(imp_->ArgsCnt() == -1 || imp_->ArgsCnt() == static_cast<int>(ranges.size()));
  imp_->Ranges(ranges);
  return this;
}

Benchmark* Benchmark::DenseRange(int start, int limit, int step) {
  CHECK(imp_->ArgsCnt() == -1 || imp_->ArgsCnt() == 1);
  imp_->DenseRange(start, limit, step);
  return this;
}

Benchmark* Benchmark::Args(const std::vector<int>& args) {
  CHECK(imp_->ArgsCnt() == -1 || imp_->ArgsCnt() == static_cast<int>(args.size()));
  imp_->Args(args);
  return this;
}

Benchmark* Benchmark::Apply(void (*custom_arguments)(Benchmark* benchmark)) {
  custom_arguments(this);
  return this;
}

Benchmark* Benchmark::RangeMultiplier(int multiplier) {
  imp_->RangeMultiplier(multiplier);
  return this;
}


Benchmark* Benchmark::Repetitions(int t) {
  imp_->Repetitions(t);
  return this;
}

Benchmark* Benchmark::MinTime(double t) {
  imp_->MinTime(t);
  return this;
}

Benchmark* Benchmark::UseRealTime() {
  imp_->UseRealTime();
  return this;
}

Benchmark* Benchmark::UseManualTime() {
  imp_->UseManualTime();
  return this;
}

Benchmark* Benchmark::Complexity(BigO complexity) {
  imp_->Complexity(complexity);
  return this;
}

Benchmark* Benchmark::Complexity(BigOFunc* complexity) {
  imp_->Complexity(oLambda);
  imp_->ComplexityLambda(complexity);
  return this;
}

Benchmark* Benchmark::Threads(int t) {
  imp_->Threads(t);
  return this;
}

Benchmark* Benchmark::ThreadRange(int min_threads, int max_threads) {
  imp_->ThreadRange(min_threads, max_threads);
  return this;
}

Benchmark* Benchmark::ThreadPerCpu() {
  imp_->ThreadPerCpu();
  return this;
}

void Benchmark::SetName(const char* name) {
  imp_->SetName(name);
}

void FunctionBenchmark::Run(State& st) {
  func_(st);
}

} // end namespace internal

namespace {

// Execute one thread of benchmark b for the specified number of iterations.
// Adds the stats collected for the thread into *total.
void RunInThread(const benchmark::internal::Benchmark::Instance* b,
                 size_t iters, int thread_id,
                 ThreadStats* total) EXCLUDES(GetBenchmarkLock()) {
  State st(iters, b->arg, thread_id, b->threads);
  b->benchmark->Run(st);
  CHECK(st.iterations() == st.max_iterations) <<
    "Benchmark returned before State::KeepRunning() returned false!";
  {
    MutexLock l(GetBenchmarkLock());
    total->bytes_processed += st.bytes_processed();
    total->items_processed += st.items_processed();
    total->complexity_n += st.complexity_length_n();
  }

  timer_manager->Finalize();
}

std::vector<BenchmarkReporter::Run>
RunBenchmark(const benchmark::internal::Benchmark::Instance& b,
             std::vector<BenchmarkReporter::Run>* complexity_reports)
  EXCLUDES(GetBenchmarkLock()) {
   std::vector<BenchmarkReporter::Run> reports; // return value
  size_t iters = 1;

  std::vector<std::thread> pool;
  if (b.multithreaded)
    pool.resize(b.threads);

  const int repeats = b.repetitions != 0 ? b.repetitions
                                         : FLAGS_benchmark_repetitions;
  for (int i = 0; i < repeats; i++) {
    std::string mem;
    for (;;) {
      // Try benchmark
      VLOG(2) << "Running " << b.name << " for " << iters << "\n";

      {
        MutexLock l(GetBenchmarkLock());
        GetReportLabel()->clear();
      }
      error_message = nullptr;

      Notification done;
      timer_manager = std::unique_ptr<TimerManager>(new TimerManager(b.threads, &done));

      ThreadStats total;
      running_benchmark = true;
      if (b.multithreaded) {
        // If this is out first iteration of the while(true) loop then the
        // threads haven't been started and can't be joined. Otherwise we need
        // to join the thread before replacing them.
        for (std::thread& thread : pool) {
          if (thread.joinable())
            thread.join();
        }
        for (std::size_t ti = 0; ti < pool.size(); ++ti) {
            pool[ti] = std::thread(&RunInThread, &b, iters, static_cast<int>(ti), &total);
        }
      } else {
        // Run directly in this thread
        RunInThread(&b, iters, 0, &total);
      }
      done.WaitForNotification();
      running_benchmark = false;

      const double cpu_accumulated_time = timer_manager->cpu_time_used();
      const double real_accumulated_time = timer_manager->real_time_used();
      const double manual_accumulated_time = timer_manager->manual_time_used();
      timer_manager.reset();

      VLOG(2) << "Ran in " << cpu_accumulated_time << "/"
              << real_accumulated_time << "\n";

      // Base decisions off of real time if requested by this benchmark.
      double seconds = cpu_accumulated_time;
      if (b.use_manual_time) {
          seconds = manual_accumulated_time;
      } else if (b.use_real_time) {
          seconds = real_accumulated_time;
      }

      std::string label;
      {
        MutexLock l(GetBenchmarkLock());
        label = *GetReportLabel();
      }
      error_message_type error_msg = error_message;

      const double min_time = !IsZero(b.min_time) ? b.min_time
                                                  : FLAGS_benchmark_min_time;

      // If this was the first run, was elapsed time or cpu time large enough?
      // If this is not the first run, go with the current value of iter.
      if ((i > 0) || (error_msg != nullptr) ||
          (iters >= kMaxIterations) ||
          (seconds >= min_time) ||
          (real_accumulated_time >= 5*min_time)) {

        // Create report about this benchmark run.
        BenchmarkReporter::Run report;
        report.benchmark_name = b.name;
        report.error_occurred = error_msg != nullptr;
        report.error_message = error_msg != nullptr ? error_msg : "";
        report.report_label = label;
        // Report the total iterations across all threads.
        report.iterations = static_cast<int64_t>(iters) * b.threads;
        report.time_unit = b.time_unit;

        if (!report.error_occurred) {
          double bytes_per_second = 0;
          if (total.bytes_processed > 0 && seconds > 0.0) {
            bytes_per_second = (total.bytes_processed / seconds);
          }
          double items_per_second = 0;
          if (total.items_processed > 0 && seconds > 0.0) {
            items_per_second = (total.items_processed / seconds);
          }

          if (b.use_manual_time) {
            report.real_accumulated_time = manual_accumulated_time;
          } else {
            report.real_accumulated_time = real_accumulated_time;
          }
          report.cpu_accumulated_time = cpu_accumulated_time;
          report.bytes_per_second = bytes_per_second;
          report.items_per_second = items_per_second;
          report.complexity_n = total.complexity_n;
          report.complexity = b.complexity;
          report.complexity_lambda = b.complexity_lambda;
          if(report.complexity != oNone)
            complexity_reports->push_back(report);
        }

        reports.push_back(report);
        break;
      }

      // See how much iterations should be increased by
      // Note: Avoid division by zero with max(seconds, 1ns).
      double multiplier = min_time * 1.4 / std::max(seconds, 1e-9);
      // If our last run was at least 10% of FLAGS_benchmark_min_time then we
      // use the multiplier directly. Otherwise we use at most 10 times
      // expansion.
      // NOTE: When the last run was at least 10% of the min time the max
      // expansion should be 14x.
      bool is_significant = (seconds / min_time) > 0.1;
      multiplier = is_significant ? multiplier : std::min(10.0, multiplier);
      if (multiplier <= 1.0) multiplier = 2.0;
      double next_iters = std::max(multiplier * iters, iters + 1.0);
      if (next_iters > kMaxIterations) {
        next_iters = kMaxIterations;
      }
      VLOG(3) << "Next iters: " << next_iters << ", " << multiplier << "\n";
      iters = static_cast<int>(next_iters + 0.5);
    }
  }
  std::vector<BenchmarkReporter::Run> additional_run_stats = ComputeStats(reports);
  reports.insert(reports.end(), additional_run_stats.begin(),
                 additional_run_stats.end());

  if((b.complexity != oNone) && b.last_benchmark_instance) {
    additional_run_stats = ComputeBigO(*complexity_reports);
    reports.insert(reports.end(), additional_run_stats.begin(),
                   additional_run_stats.end());
    complexity_reports->clear();
  }

  if (b.multithreaded) {
    for (std::thread& thread : pool)
      thread.join();
  }

  return reports;
}

}  // namespace

State::State(size_t max_iters, const std::vector<int>& ranges,
             int thread_i, int n_threads)
    : started_(false), finished_(false), total_iterations_(0),
      range_(ranges),
      bytes_processed_(0), items_processed_(0),
      complexity_n_(0),
      error_occurred_(false),
      thread_index(thread_i),
      threads(n_threads),
      max_iterations(max_iters)
{
    CHECK(max_iterations != 0) << "At least one iteration must be run";
    CHECK_LT(thread_index, threads) << "thread_index must be less than threads";
}

void State::PauseTiming() {
  // Add in time accumulated so far
  CHECK(running_benchmark);
  CHECK(started_ && !finished_ && !error_occurred_);
  timer_manager->StopTimer();
}

void State::ResumeTiming() {
  CHECK(running_benchmark);
  CHECK(started_ && !finished_ && !error_occurred_);
  timer_manager->StartTimer();
}

void State::SkipWithError(const char* msg) {
  CHECK(msg);
  error_occurred_ = true;
  error_message_type expected_no_error_msg = nullptr;
  error_message.compare_exchange_weak(expected_no_error_msg,
    const_cast<error_message_type>(msg));
  started_ = finished_ = true;
  total_iterations_ = max_iterations;
  timer_manager->RemoveErroredThread();
}

void State::SetIterationTime(double seconds)
{
  CHECK(running_benchmark);
  timer_manager->SetIterationTime(seconds);
}

void State::SetLabel(const char* label) {
  CHECK(running_benchmark);
  MutexLock l(GetBenchmarkLock());
  *GetReportLabel() = label;
}

namespace internal {
namespace {

void RunMatchingBenchmarks(const std::vector<Benchmark::Instance>& benchmarks,
                           BenchmarkReporter* console_reporter,
                           BenchmarkReporter* file_reporter) {
  // Note the file_reporter can be null.
  CHECK(console_reporter != nullptr);

  // Determine the width of the name field using a minimum width of 10.
  bool has_repetitions = FLAGS_benchmark_repetitions > 1;
  size_t name_field_width = 10;
  for (const Benchmark::Instance& benchmark : benchmarks) {
    name_field_width =
        std::max<size_t>(name_field_width, benchmark.name.size());
    has_repetitions |= benchmark.repetitions > 1;
  }
  if (has_repetitions)
    name_field_width += std::strlen("_stddev");

  // Print header here
  BenchmarkReporter::Context context;
  context.num_cpus = NumCPUs();
  context.mhz_per_cpu = CyclesPerSecond() / 1000000.0f;

  context.cpu_scaling_enabled = CpuScalingEnabled();
  context.name_field_width = name_field_width;

  // Keep track of runing times of all instances of current benchmark
  std::vector<BenchmarkReporter::Run> complexity_reports;

  if (console_reporter->ReportContext(context)
      && (!file_reporter || file_reporter->ReportContext(context))) {
    for (const auto& benchmark : benchmarks) {
      std::vector<BenchmarkReporter::Run> reports =
          RunBenchmark(benchmark, &complexity_reports);
      console_reporter->ReportRuns(reports);
      if (file_reporter) file_reporter->ReportRuns(reports);
    }
  }
  console_reporter->Finalize();
  if (file_reporter) file_reporter->Finalize();
}

std::unique_ptr<BenchmarkReporter>
CreateReporter(std::string const& name, ConsoleReporter::OutputOptions allow_color) {
  typedef std::unique_ptr<BenchmarkReporter> PtrType;
  if (name == "console") {
    return PtrType(new ConsoleReporter(allow_color));
  } else if (name == "json") {
    return PtrType(new JSONReporter);
  } else if (name == "csv") {
    return PtrType(new CSVReporter);
  } else {
    std::cerr << "Unexpected format: '" << name << "'\n";
    std::exit(1);
  }
}

} // end namespace
} // end namespace internal

size_t RunSpecifiedBenchmarks() {
  return RunSpecifiedBenchmarks(nullptr, nullptr);
}


size_t RunSpecifiedBenchmarks(BenchmarkReporter* console_reporter) {
  return RunSpecifiedBenchmarks(console_reporter, nullptr);
}


size_t RunSpecifiedBenchmarks(BenchmarkReporter* console_reporter,
                              BenchmarkReporter* file_reporter) {
  std::string spec = FLAGS_benchmark_filter;
  if (spec.empty() || spec == "all")
    spec = ".";  // Regexp that matches all benchmarks

  std::vector<internal::Benchmark::Instance> benchmarks;
  auto families = internal::BenchmarkFamilies::GetInstance();
  if (!families->FindBenchmarks(spec, &benchmarks)) return 0;

  if (FLAGS_benchmark_list_tests) {
    for (auto const& benchmark : benchmarks)
      std::cout <<  benchmark.name << "\n";
  } else {
    // Setup the reporters
    std::ofstream output_file;
    std::unique_ptr<BenchmarkReporter> default_console_reporter;
    std::unique_ptr<BenchmarkReporter> default_file_reporter;
    if (!console_reporter) {
      auto output_opts = FLAGS_color_print ? ConsoleReporter::OO_Color
                                           : ConsoleReporter::OO_None;
      default_console_reporter = internal::CreateReporter(
          FLAGS_benchmark_format, output_opts);
      console_reporter = default_console_reporter.get();
    }
    std::string const& fname = FLAGS_benchmark_out;
    if (fname == "" && file_reporter) {
      std::cerr << "A custom file reporter was provided but "
                   "--benchmark_out=<file> was not specified." << std::endl;
      std::exit(1);
    }
    if (fname != "") {
      output_file.open(fname);
      if (!output_file.is_open()) {
        std::cerr << "invalid file name: '" << fname << std::endl;
        std::exit(1);
      }
      if (!file_reporter) {
        default_file_reporter = internal::CreateReporter(
            FLAGS_benchmark_out_format, ConsoleReporter::OO_None);
        file_reporter = default_file_reporter.get();
      }
      file_reporter->SetOutputStream(&output_file);
      file_reporter->SetErrorStream(&output_file);
    }
    internal::RunMatchingBenchmarks(benchmarks, console_reporter, file_reporter);
  }
  return benchmarks.size();
}

namespace internal {

void PrintUsageAndExit() {
  fprintf(stdout,
          "benchmark"
          " [--benchmark_list_tests={true|false}]\n"
          "          [--benchmark_filter=<regex>]\n"
          "          [--benchmark_min_time=<min_time>]\n"
          "          [--benchmark_repetitions=<num_repetitions>]\n"
          "          [--benchmark_format=<console|json|csv>]\n"
          "          [--benchmark_out=<filename>]\n"
          "          [--benchmark_out_format=<json|console|csv>]\n"
          "          [--color_print={true|false}]\n"
          "          [--v=<verbosity>]\n");
  exit(0);
}

void ParseCommandLineFlags(int* argc, char** argv) {
  using namespace benchmark;
  for (int i = 1; i < *argc; ++i) {
    if (
        ParseBoolFlag(argv[i], "benchmark_list_tests",
                      &FLAGS_benchmark_list_tests) ||
        ParseStringFlag(argv[i], "benchmark_filter",
                        &FLAGS_benchmark_filter) ||
        ParseDoubleFlag(argv[i], "benchmark_min_time",
                        &FLAGS_benchmark_min_time) ||
        ParseInt32Flag(argv[i], "benchmark_repetitions",
                       &FLAGS_benchmark_repetitions) ||
        ParseStringFlag(argv[i], "benchmark_format",
                        &FLAGS_benchmark_format) ||
        ParseStringFlag(argv[i], "benchmark_out",
                        &FLAGS_benchmark_out) ||
        ParseStringFlag(argv[i], "benchmark_out_format",
                        &FLAGS_benchmark_out_format) ||
        ParseBoolFlag(argv[i], "color_print",
                       &FLAGS_color_print) ||
        ParseInt32Flag(argv[i], "v", &FLAGS_v)) {
      for (int j = i; j != *argc; ++j) argv[j] = argv[j + 1];

      --(*argc);
      --i;
    } else if (IsFlag(argv[i], "help")) {
      PrintUsageAndExit();
    }
  }
  for (auto const* flag : {&FLAGS_benchmark_format,
                           &FLAGS_benchmark_out_format})
  if (*flag != "console" && *flag != "json" && *flag != "csv") {
    PrintUsageAndExit();
  }
}

Benchmark* RegisterBenchmarkInternal(Benchmark* bench) {
    std::unique_ptr<Benchmark> bench_ptr(bench);
    BenchmarkFamilies* families = BenchmarkFamilies::GetInstance();
    families->AddBenchmark(std::move(bench_ptr));
    return bench;
}

} // end namespace internal

void Initialize(int* argc, char** argv) {
  internal::ParseCommandLineFlags(argc, argv);
  internal::SetLogLevel(FLAGS_v);
  // TODO remove this. It prints some output the first time it is called.
  // We don't want to have this ouput printed during benchmarking.
  MyCPUUsage();
  // The first call to walltime::Now initialized it. Call it once to
  // prevent the initialization from happening in a benchmark.
  walltime::Now();
}

} // end namespace benchmark
