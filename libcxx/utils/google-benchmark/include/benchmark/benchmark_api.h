// Support for registering benchmarks for functions.

/* Example usage:
// Define a function that executes the code to be measured a
// specified number of times:
static void BM_StringCreation(benchmark::State& state) {
  while (state.KeepRunning())
    std::string empty_string;
}

// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  while (state.KeepRunning())
    std::string copy(x);
}
BENCHMARK(BM_StringCopy);

// Augment the main() program to invoke benchmarks if specified
// via the --benchmarks command line flag.  E.g.,
//       my_unittest --benchmark_filter=all
//       my_unittest --benchmark_filter=BM_StringCreation
//       my_unittest --benchmark_filter=String
//       my_unittest --benchmark_filter='Copy|Creation'
int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}

// Sometimes a family of microbenchmarks can be implemented with
// just one routine that takes an extra argument to specify which
// one of the family of benchmarks to run.  For example, the following
// code defines a family of microbenchmarks for measuring the speed
// of memcpy() calls of different lengths:

static void BM_memcpy(benchmark::State& state) {
  char* src = new char[state.range_x()]; char* dst = new char[state.range_x()];
  memset(src, 'x', state.range_x());
  while (state.KeepRunning())
    memcpy(dst, src, state.range_x());
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range_x()));
  delete[] src; delete[] dst;
}
BENCHMARK(BM_memcpy)->Arg(8)->Arg(64)->Arg(512)->Arg(1<<10)->Arg(8<<10);

// The preceding code is quite repetitive, and can be replaced with the
// following short-hand.  The following invocation will pick a few
// appropriate arguments in the specified range and will generate a
// microbenchmark for each such argument.
BENCHMARK(BM_memcpy)->Range(8, 8<<10);

// You might have a microbenchmark that depends on two inputs.  For
// example, the following code defines a family of microbenchmarks for
// measuring the speed of set insertion.
static void BM_SetInsert(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    set<int> data = ConstructRandomSet(state.range_x());
    state.ResumeTiming();
    for (int j = 0; j < state.range_y(); ++j)
      data.insert(RandomNumber());
  }
}
BENCHMARK(BM_SetInsert)
   ->ArgPair(1<<10, 1)
   ->ArgPair(1<<10, 8)
   ->ArgPair(1<<10, 64)
   ->ArgPair(1<<10, 512)
   ->ArgPair(8<<10, 1)
   ->ArgPair(8<<10, 8)
   ->ArgPair(8<<10, 64)
   ->ArgPair(8<<10, 512);

// The preceding code is quite repetitive, and can be replaced with
// the following short-hand.  The following macro will pick a few
// appropriate arguments in the product of the two specified ranges
// and will generate a microbenchmark for each such pair.
BENCHMARK(BM_SetInsert)->RangePair(1<<10, 8<<10, 1, 512);

// For more complex patterns of inputs, passing a custom function
// to Apply allows programmatic specification of an
// arbitrary set of arguments to run the microbenchmark on.
// The following example enumerates a dense range on
// one parameter, and a sparse range on the second.
static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i = 0; i <= 10; ++i)
    for (int j = 32; j <= 1024*1024; j *= 8)
      b->ArgPair(i, j);
}
BENCHMARK(BM_SetInsert)->Apply(CustomArguments);

// Templated microbenchmarks work the same way:
// Produce then consume 'size' messages 'iters' times
// Measures throughput in the absence of multiprogramming.
template <class Q> int BM_Sequential(benchmark::State& state) {
  Q q;
  typename Q::value_type v;
  while (state.KeepRunning()) {
    for (int i = state.range_x(); i--; )
      q.push(v);
    for (int e = state.range_x(); e--; )
      q.Wait(&v);
  }
  // actually messages, not bytes:
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations())*state.range_x());
}
BENCHMARK_TEMPLATE(BM_Sequential, WaitQueue<int>)->Range(1<<0, 1<<10);

Use `Benchmark::MinTime(double t)` to set the minimum time used to run the
benchmark. This option overrides the `benchmark_min_time` flag.

void BM_test(benchmark::State& state) {
 ... body ...
}
BENCHMARK(BM_test)->MinTime(2.0); // Run for at least 2 seconds.

In a multithreaded test, it is guaranteed that none of the threads will start
until all have called KeepRunning, and all will have finished before KeepRunning
returns false. As such, any global setup or teardown you want to do can be
wrapped in a check against the thread index:

static void BM_MultiThreaded(benchmark::State& state) {
  if (state.thread_index == 0) {
    // Setup code here.
  }
  while (state.KeepRunning()) {
    // Run the test as normal.
  }
  if (state.thread_index == 0) {
    // Teardown code here.
  }
}
BENCHMARK(BM_MultiThreaded)->Threads(4);


If a benchmark runs a few milliseconds it may be hard to visually compare the
measured times, since the output data is given in nanoseconds per default. In
order to manually set the time unit, you can specify it manually:

BENCHMARK(BM_test)->Unit(benchmark::kMillisecond);
*/

#ifndef BENCHMARK_BENCHMARK_API_H_
#define BENCHMARK_BENCHMARK_API_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "macros.h"

namespace benchmark {
class BenchmarkReporter;

void Initialize(int* argc, char** argv);

// Generate a list of benchmarks matching the specified --benchmark_filter flag
// and if --benchmark_list_tests is specified return after printing the name
// of each matching benchmark. Otherwise run each matching benchmark and
// report the results.
//
// The second overload reports the results using the specified 'reporter'.
//
// RETURNS: The number of matching benchmarks.
size_t RunSpecifiedBenchmarks();
size_t RunSpecifiedBenchmarks(BenchmarkReporter* reporter);


// If this routine is called, peak memory allocation past this point in the
// benchmark is reported at the end of the benchmark report line. (It is
// computed by running the benchmark once with a single iteration and a memory
// tracer.)
// TODO(dominic)
// void MemoryUsage();

namespace internal {
class Benchmark;
class BenchmarkImp;
class BenchmarkFamilies;

template <class T> struct Voider {
    typedef void type;
};

template <class T, class = void>
struct EnableIfString {};

template <class T>
struct EnableIfString<T, typename Voider<typename T::basic_string>::type> {
    typedef int type;
};

void UseCharPointer(char const volatile*);

// Take ownership of the pointer and register the benchmark. Return the
// registered benchmark.
Benchmark* RegisterBenchmarkInternal(Benchmark*);

} // end namespace internal


// The DoNotOptimize(...) function can be used to prevent a value or
// expression from being optimized away by the compiler. This function is
// intended to add little to no overhead.
// See: https://youtu.be/nXaxk27zwlk?t=2441
#if defined(__GNUC__)
template <class Tp>
inline BENCHMARK_ALWAYS_INLINE void DoNotOptimize(Tp const& value) {
    asm volatile("" : : "g"(value) : "memory");
}
// Force the compiler to flush pending writes to global memory. Acts as an
// effective read/write barrier
inline BENCHMARK_ALWAYS_INLINE void ClobberMemory() {
    asm volatile("" : : : "memory");
}
#else
template <class Tp>
inline BENCHMARK_ALWAYS_INLINE void DoNotOptimize(Tp const& value) {
    internal::UseCharPointer(&reinterpret_cast<char const volatile&>(value));
}
// FIXME Add ClobberMemory() for non-gnu compilers
#endif

// TimeUnit is passed to a benchmark in order to specify the order of magnitude
// for the measured time.
enum TimeUnit {
  kNanosecond,
  kMicrosecond,
  kMillisecond
};

// BigO is passed to a benchmark in order to specify the asymptotic computational 
// complexity for the benchmark. In case oAuto is selected, complexity will be 
// calculated automatically to the best fit.
enum BigO {
  oNone,
  o1,
  oN,
  oNSquared,
  oNCubed,
  oLogN,
  oNLogN,
  oAuto,
  oLambda
};

// BigOFunc is passed to a benchmark in order to specify the asymptotic 
// computational complexity for the benchmark.
typedef double(BigOFunc)(int);

// State is passed to a running Benchmark and contains state for the
// benchmark to use.
class State {
public:
  State(size_t max_iters, bool has_x, int x, bool has_y, int y,
        int thread_i, int n_threads);

  // Returns true if the benchmark should continue through another iteration.
  // NOTE: A benchmark may not return from the test until KeepRunning() has
  // returned false.
  bool KeepRunning() {
    if (BENCHMARK_BUILTIN_EXPECT(!started_, false)) {
      assert(!finished_);
      started_ = true;
      ResumeTiming();
    }
    bool const res = total_iterations_++ < max_iterations;
    if (BENCHMARK_BUILTIN_EXPECT(!res, false)) {
      assert(started_ && (!finished_ || error_occurred_));
      if (!error_occurred_) {
        PauseTiming();
      }
      // Total iterations now is one greater than max iterations. Fix this.
      total_iterations_ = max_iterations;
      finished_ = true;
    }
    return res;
  }

  // REQUIRES: timer is running and 'SkipWithError(...)' has not been called
  //           by the current thread.
  // Stop the benchmark timer.  If not called, the timer will be
  // automatically stopped after KeepRunning() returns false for the first time.
  //
  // For threaded benchmarks the PauseTiming() function acts
  // like a barrier.  I.e., the ith call by a particular thread to this
  // function will block until all active threads have made their ith call.
  // The timer will stop when the last thread has called this function.
  //
  // NOTE: PauseTiming()/ResumeTiming() are relatively
  // heavyweight, and so their use should generally be avoided
  // within each benchmark iteration, if possible.
  void PauseTiming();

  // REQUIRES: timer is not running and 'SkipWithError(...)' has not been called
  //           by the current thread.
  // Start the benchmark timer.  The timer is NOT running on entrance to the
  // benchmark function. It begins running after the first call to KeepRunning()
  //
  // For threaded benchmarks the ResumeTiming() function acts
  // like a barrier.  I.e., the ith call by a particular thread to this
  // function will block until all active threads have made their ith call.
  // The timer will start when the last thread has called this function.
  //
  // NOTE: PauseTiming()/ResumeTiming() are relatively
  // heavyweight, and so their use should generally be avoided
  // within each benchmark iteration, if possible.
  void ResumeTiming();

  // REQUIRES: 'SkipWithError(...)' has not been called previously by the
  //            current thread.
  // Skip any future iterations of the 'KeepRunning()' loop in the current
  // thread and report an error with the specified 'msg'. After this call
  // the user may explicitly 'return' from the benchmark.
  //
  // For threaded benchmarks only the current thread stops executing. If
  // multiple threads report an error only the first error message is used.
  // The current thread is no longer considered 'active' by
  // 'PauseTiming()' and 'ResumingTiming()'.
  //
  // NOTE: Calling 'SkipWithError(...)' does not cause the benchmark to exit
  // the current scope immediately. If the function is called from within
  // the 'KeepRunning()' loop the current iteration will finish. It is the users
  // responsibility to exit the scope as needed.
  void SkipWithError(const char* msg);

  // REQUIRES: called exactly once per iteration of the KeepRunning loop.
  // Set the manually measured time for this benchmark iteration, which
  // is used instead of automatically measured time if UseManualTime() was
  // specified.
  //
  // For threaded benchmarks the SetIterationTime() function acts
  // like a barrier.  I.e., the ith call by a particular thread to this
  // function will block until all threads have made their ith call.
  // The time will be set by the last thread to call this function.
  void SetIterationTime(double seconds);

  // Set the number of bytes processed by the current benchmark
  // execution.  This routine is typically called once at the end of a
  // throughput oriented benchmark.  If this routine is called with a
  // value > 0, the report is printed in MB/sec instead of nanoseconds
  // per iteration.
  //
  // REQUIRES: a benchmark has exited its KeepRunning loop.
  BENCHMARK_ALWAYS_INLINE
  void SetBytesProcessed(size_t bytes) {
    bytes_processed_ = bytes;
  }

  BENCHMARK_ALWAYS_INLINE
  size_t bytes_processed() const {
    return bytes_processed_;
  }

  // If this routine is called with complexity_n > 0 and complexity report is requested for the 
  // family benchmark, then current benchmark will be part of the computation and complexity_n will
  // represent the length of N.
  BENCHMARK_ALWAYS_INLINE
  void SetComplexityN(int complexity_n) {
    complexity_n_ = complexity_n;
  }

  BENCHMARK_ALWAYS_INLINE
  size_t complexity_length_n() {
    return complexity_n_;
  }

  // If this routine is called with items > 0, then an items/s
  // label is printed on the benchmark report line for the currently
  // executing benchmark. It is typically called at the end of a processing
  // benchmark where a processing items/second output is desired.
  //
  // REQUIRES: a benchmark has exited its KeepRunning loop.
  BENCHMARK_ALWAYS_INLINE
  void SetItemsProcessed(size_t items) {
    items_processed_ = items;
  }

  BENCHMARK_ALWAYS_INLINE
  size_t items_processed() const {
    return items_processed_;
  }

  // If this routine is called, the specified label is printed at the
  // end of the benchmark report line for the currently executing
  // benchmark.  Example:
  //  static void BM_Compress(benchmark::State& state) {
  //    ...
  //    double compress = input_size / output_size;
  //    state.SetLabel(StringPrintf("compress:%.1f%%", 100.0*compression));
  //  }
  // Produces output that looks like:
  //  BM_Compress   50         50   14115038  compress:27.3%
  //
  // REQUIRES: a benchmark has exited its KeepRunning loop.
  void SetLabel(const char* label);

  // Allow the use of std::string without actually including <string>.
  // This function does not participate in overload resolution unless StringType
  // has the nested typename `basic_string`. This typename should be provided
  // as an injected class name in the case of std::string.
  template <class StringType>
  void SetLabel(StringType const & str,
                typename internal::EnableIfString<StringType>::type = 1) {
    this->SetLabel(str.c_str());
  }

  // Range arguments for this run. CHECKs if the argument has been set.
  BENCHMARK_ALWAYS_INLINE
  int range_x() const {
    assert(has_range_x_);
    ((void)has_range_x_); // Prevent unused warning.
    return range_x_;
  }

  BENCHMARK_ALWAYS_INLINE
  int range_y() const {
    assert(has_range_y_);
    ((void)has_range_y_); // Prevent unused warning.
    return range_y_;
  }

  BENCHMARK_ALWAYS_INLINE
  size_t iterations() const { return total_iterations_; }

private:
  bool started_;
  bool finished_;
  size_t total_iterations_;

  bool has_range_x_;
  int range_x_;

  bool has_range_y_;
  int range_y_;

  size_t bytes_processed_;
  size_t items_processed_;

  int complexity_n_;

public:
  // FIXME: Make this private somehow.
  bool error_occurred_;
public:
  // Index of the executing thread. Values from [0, threads).
  const int thread_index;
  // Number of threads concurrently executing the benchmark.
  const int threads;
  const size_t max_iterations;

private:
  BENCHMARK_DISALLOW_COPY_AND_ASSIGN(State);
};

namespace internal {

typedef void(Function)(State&);

// ------------------------------------------------------
// Benchmark registration object.  The BENCHMARK() macro expands
// into an internal::Benchmark* object.  Various methods can
// be called on this object to change the properties of the benchmark.
// Each method returns "this" so that multiple method calls can
// chained into one expression.
class Benchmark {
public:
  virtual ~Benchmark();

  // Note: the following methods all return "this" so that multiple
  // method calls can be chained together in one expression.

  // Run this benchmark once with "x" as the extra argument passed
  // to the function.
  // REQUIRES: The function passed to the constructor must accept an arg1.
  Benchmark* Arg(int x);

  // Run this benchmark with the given time unit for the generated output report
  Benchmark* Unit(TimeUnit unit);

  // Run this benchmark once for a number of values picked from the
  // range [start..limit].  (start and limit are always picked.)
  // REQUIRES: The function passed to the constructor must accept an arg1.
  Benchmark* Range(int start, int limit);

  // Run this benchmark once for every value in the range [start..limit]
  // REQUIRES: The function passed to the constructor must accept an arg1.
  Benchmark* DenseRange(int start, int limit);

  // Run this benchmark once with "x,y" as the extra arguments passed
  // to the function.
  // REQUIRES: The function passed to the constructor must accept arg1,arg2.
  Benchmark* ArgPair(int x, int y);

  // Pick a set of values A from the range [lo1..hi1] and a set
  // of values B from the range [lo2..hi2].  Run the benchmark for
  // every pair of values in the cartesian product of A and B
  // (i.e., for all combinations of the values in A and B).
  // REQUIRES: The function passed to the constructor must accept arg1,arg2.
  Benchmark* RangePair(int lo1, int hi1, int lo2, int hi2);

  // Pass this benchmark object to *func, which can customize
  // the benchmark by calling various methods like Arg, ArgPair,
  // Threads, etc.
  Benchmark* Apply(void (*func)(Benchmark* benchmark));

  // Set the range multiplier for non-dense range. If not called, the range multiplier 
  // kRangeMultiplier will be used.
  Benchmark* RangeMultiplier(int multiplier);

  // Set the minimum amount of time to use when running this benchmark. This
  // option overrides the `benchmark_min_time` flag.
  // REQUIRES: `t > 0`
  Benchmark* MinTime(double t);

  // Specify the amount of times to repeat this benchmark. This option overrides
  // the `benchmark_repetitions` flag.
  // REQUIRES: `n > 0`
  Benchmark* Repetitions(int n);

  // If a particular benchmark is I/O bound, runs multiple threads internally or
  // if for some reason CPU timings are not representative, call this method. If
  // called, the elapsed time will be used to control how many iterations are
  // run, and in the printing of items/second or MB/seconds values.  If not
  // called, the cpu time used by the benchmark will be used.
  Benchmark* UseRealTime();

  // If a benchmark must measure time manually (e.g. if GPU execution time is being
  // measured), call this method. If called, each benchmark iteration should call
  // SetIterationTime(seconds) to report the measured time, which will be used
  // to control how many iterations are run, and in the printing of items/second
  // or MB/second values.
  Benchmark* UseManualTime();

  // Set the asymptotic computational complexity for the benchmark. If called
  // the asymptotic computational complexity will be shown on the output. 
  Benchmark* Complexity(BigO complexity = benchmark::oAuto);

  // Set the asymptotic computational complexity for the benchmark. If called
  // the asymptotic computational complexity will be shown on the output.
  Benchmark* Complexity(BigOFunc* complexity);

  // Support for running multiple copies of the same benchmark concurrently
  // in multiple threads.  This may be useful when measuring the scaling
  // of some piece of code.

  // Run one instance of this benchmark concurrently in t threads.
  Benchmark* Threads(int t);

  // Pick a set of values T from [min_threads,max_threads].
  // min_threads and max_threads are always included in T.  Run this
  // benchmark once for each value in T.  The benchmark run for a
  // particular value t consists of t threads running the benchmark
  // function concurrently.  For example, consider:
  //    BENCHMARK(Foo)->ThreadRange(1,16);
  // This will run the following benchmarks:
  //    Foo in 1 thread
  //    Foo in 2 threads
  //    Foo in 4 threads
  //    Foo in 8 threads
  //    Foo in 16 threads
  Benchmark* ThreadRange(int min_threads, int max_threads);

  // Equivalent to ThreadRange(NumCPUs(), NumCPUs())
  Benchmark* ThreadPerCpu();

  virtual void Run(State& state) = 0;

  // Used inside the benchmark implementation
  struct Instance;

protected:
  explicit Benchmark(const char* name);
  Benchmark(Benchmark const&);
  void SetName(const char* name);

private:
  friend class BenchmarkFamilies;
  BenchmarkImp* imp_;

  Benchmark& operator=(Benchmark const&);
};

// The class used to hold all Benchmarks created from static function.
// (ie those created using the BENCHMARK(...) macros.
class FunctionBenchmark : public Benchmark {
public:
    FunctionBenchmark(const char* name, Function* func)
        : Benchmark(name), func_(func)
    {}

    virtual void Run(State& st);
private:
    Function* func_;
};

}  // end namespace internal

// The base class for all fixture tests.
class Fixture: public internal::Benchmark {
public:
    Fixture() : internal::Benchmark("") {}

    virtual void Run(State& st) {
      this->SetUp(st);
      this->BenchmarkCase(st);
      this->TearDown(st);
    }

    virtual void SetUp(const State&) {}
    virtual void TearDown(const State&) {}

protected:
    virtual void BenchmarkCase(State&) = 0;
};

}  // end namespace benchmark


// ------------------------------------------------------
// Macro to register benchmarks

// Check that __COUNTER__ is defined and that __COUNTER__ increases by 1
// every time it is expanded. X + 1 == X + 0 is used in case X is defined to be
// empty. If X is empty the expression becomes (+1 == +0).
#if defined(__COUNTER__) && (__COUNTER__ + 1 == __COUNTER__ + 0)
#define BENCHMARK_PRIVATE_UNIQUE_ID __COUNTER__
#else
#define BENCHMARK_PRIVATE_UNIQUE_ID __LINE__
#endif

// Helpers for generating unique variable names
#define BENCHMARK_PRIVATE_NAME(n) \
    BENCHMARK_PRIVATE_CONCAT(_benchmark_, BENCHMARK_PRIVATE_UNIQUE_ID, n)
#define BENCHMARK_PRIVATE_CONCAT(a, b, c) BENCHMARK_PRIVATE_CONCAT2(a, b, c)
#define BENCHMARK_PRIVATE_CONCAT2(a, b, c) a##b##c

#define BENCHMARK_PRIVATE_DECLARE(n)       \
  static ::benchmark::internal::Benchmark* \
  BENCHMARK_PRIVATE_NAME(n) BENCHMARK_UNUSED

#define BENCHMARK(n) \
    BENCHMARK_PRIVATE_DECLARE(n) =                               \
        (::benchmark::internal::RegisterBenchmarkInternal(       \
            new ::benchmark::internal::FunctionBenchmark(#n, n)))

// Old-style macros
#define BENCHMARK_WITH_ARG(n, a) BENCHMARK(n)->Arg((a))
#define BENCHMARK_WITH_ARG2(n, a1, a2) BENCHMARK(n)->ArgPair((a1), (a2))
#define BENCHMARK_WITH_UNIT(n, t) BENCHMARK(n)->Unit((t))
#define BENCHMARK_RANGE(n, lo, hi) BENCHMARK(n)->Range((lo), (hi))
#define BENCHMARK_RANGE2(n, l1, h1, l2, h2) \
  BENCHMARK(n)->RangePair((l1), (h1), (l2), (h2))

#if __cplusplus >= 201103L

// Register a benchmark which invokes the function specified by `func`
// with the additional arguments specified by `...`.
//
// For example:
//
// template <class ...ExtraArgs>`
// void BM_takes_args(benchmark::State& state, ExtraArgs&&... extra_args) {
//  [...]
//}
// /* Registers a benchmark named "BM_takes_args/int_string_test` */
// BENCHMARK_CAPTURE(BM_takes_args, int_string_test, 42, std::string("abc"));
#define BENCHMARK_CAPTURE(func, test_case_name, ...)                       \
    BENCHMARK_PRIVATE_DECLARE(func) =                                      \
        (::benchmark::internal::RegisterBenchmarkInternal(                 \
            new ::benchmark::internal::FunctionBenchmark(                  \
                    #func "/" #test_case_name,                             \
                    [](::benchmark::State& st) { func(st, __VA_ARGS__); })))

#endif // __cplusplus >= 11

// This will register a benchmark for a templatized function.  For example:
//
// template<int arg>
// void BM_Foo(int iters);
//
// BENCHMARK_TEMPLATE(BM_Foo, 1);
//
// will register BM_Foo<1> as a benchmark.
#define BENCHMARK_TEMPLATE1(n, a) \
  BENCHMARK_PRIVATE_DECLARE(n) =  \
      (::benchmark::internal::RegisterBenchmarkInternal( \
        new ::benchmark::internal::FunctionBenchmark(#n "<" #a ">", n<a>)))

#define BENCHMARK_TEMPLATE2(n, a, b)                     \
  BENCHMARK_PRIVATE_DECLARE(n) =                         \
      (::benchmark::internal::RegisterBenchmarkInternal( \
        new ::benchmark::internal::FunctionBenchmark(    \
            #n "<" #a "," #b ">", n<a, b>)))

#if __cplusplus >= 201103L
#define BENCHMARK_TEMPLATE(n, ...)           \
  BENCHMARK_PRIVATE_DECLARE(n) =             \
      (::benchmark::internal::RegisterBenchmarkInternal( \
        new ::benchmark::internal::FunctionBenchmark( \
        #n "<" #__VA_ARGS__ ">", n<__VA_ARGS__>)))
#else
#define BENCHMARK_TEMPLATE(n, a) BENCHMARK_TEMPLATE1(n, a)
#endif


#define BENCHMARK_PRIVATE_DECLARE_F(BaseClass, Method)      \
class BaseClass##_##Method##_Benchmark : public BaseClass { \
public:                                                     \
    BaseClass##_##Method##_Benchmark() : BaseClass() {      \
        this->SetName(#BaseClass "/" #Method);}             \
protected:                                                  \
    virtual void BenchmarkCase(::benchmark::State&);        \
};

#define BENCHMARK_DEFINE_F(BaseClass, Method) \
    BENCHMARK_PRIVATE_DECLARE_F(BaseClass, Method) \
    void BaseClass##_##Method##_Benchmark::BenchmarkCase

#define BENCHMARK_REGISTER_F(BaseClass, Method) \
    BENCHMARK_PRIVATE_REGISTER_F(BaseClass##_##Method##_Benchmark)

#define BENCHMARK_PRIVATE_REGISTER_F(TestName) \
    BENCHMARK_PRIVATE_DECLARE(TestName) = \
        (::benchmark::internal::RegisterBenchmarkInternal(new TestName()))

// This macro will define and register a benchmark within a fixture class.
#define BENCHMARK_F(BaseClass, Method) \
    BENCHMARK_PRIVATE_DECLARE_F(BaseClass, Method) \
    BENCHMARK_REGISTER_F(BaseClass, Method); \
    void BaseClass##_##Method##_Benchmark::BenchmarkCase


// Helper macro to create a main routine in a test that runs the benchmarks
#define BENCHMARK_MAIN()                   \
  int main(int argc, char** argv) {        \
    ::benchmark::Initialize(&argc, argv);  \
    ::benchmark::RunSpecifiedBenchmarks(); \
  }

#endif  // BENCHMARK_BENCHMARK_API_H_
