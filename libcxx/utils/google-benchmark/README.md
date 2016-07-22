# benchmark
[![Build Status](https://travis-ci.org/google/benchmark.svg?branch=master)](https://travis-ci.org/google/benchmark)
[![Build status](https://ci.appveyor.com/api/projects/status/u0qsyp7t1tk7cpxs/branch/master?svg=true)](https://ci.appveyor.com/project/google/benchmark/branch/master)
[![Coverage Status](https://coveralls.io/repos/google/benchmark/badge.svg)](https://coveralls.io/r/google/benchmark)

A library to support the benchmarking of functions, similar to unit-tests.

Discussion group: https://groups.google.com/d/forum/benchmark-discuss

IRC channel: https://freenode.net #googlebenchmark

## Example usage
### Basic usage
Define a function that executes the code to be measured.

```c++
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

BENCHMARK_MAIN();
```

### Passing arguments
Sometimes a family of benchmarks can be implemented with just one routine that
takes an extra argument to specify which one of the family of benchmarks to
run. For example, the following code defines a family of benchmarks for
measuring the speed of `memcpy()` calls of different lengths:

```c++
static void BM_memcpy(benchmark::State& state) {
  char* src = new char[state.range_x()];
  char* dst = new char[state.range_x()];
  memset(src, 'x', state.range_x());
  while (state.KeepRunning())
    memcpy(dst, src, state.range_x());
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range_x()));
  delete[] src;
  delete[] dst;
}
BENCHMARK(BM_memcpy)->Arg(8)->Arg(64)->Arg(512)->Arg(1<<10)->Arg(8<<10);
```

The preceding code is quite repetitive, and can be replaced with the following
short-hand. The following invocation will pick a few appropriate arguments in
the specified range and will generate a benchmark for each such argument.

```c++
BENCHMARK(BM_memcpy)->Range(8, 8<<10);
```

By default the arguments in the range are generated in multiples of eight and
the command above selects [ 8, 64, 512, 4k, 8k ]. In the following code the
range multiplier is changed to multiples of two.

```c++
BENCHMARK(BM_memcpy)->RangeMultiplier(2)->Range(8, 8<<10);
```
Now arguments generated are [ 8, 16, 32, 64, 128, 256, 512, 1024, 2k, 4k, 8k ].

You might have a benchmark that depends on two inputs. For example, the
following code defines a family of benchmarks for measuring the speed of set
insertion.

```c++
static void BM_SetInsert(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    std::set<int> data = ConstructRandomSet(state.range_x());
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
```

The preceding code is quite repetitive, and can be replaced with the following
short-hand. The following macro will pick a few appropriate arguments in the
product of the two specified ranges and will generate a benchmark for each such
pair.

```c++
BENCHMARK(BM_SetInsert)->RangePair(1<<10, 8<<10, 1, 512);
```

For more complex patterns of inputs, passing a custom function to `Apply` allows
programmatic specification of an arbitrary set of arguments on which to run the
benchmark. The following example enumerates a dense range on one parameter,
and a sparse range on the second.

```c++
static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i = 0; i <= 10; ++i)
    for (int j = 32; j <= 1024*1024; j *= 8)
      b->ArgPair(i, j);
}
BENCHMARK(BM_SetInsert)->Apply(CustomArguments);
```

### Calculate asymptotic complexity (Big O)
Asymptotic complexity might be calculated for a family of benchmarks. The
following code will calculate the coefficient for the high-order term in the
running time and the normalized root-mean square error of string comparison.

```c++
static void BM_StringCompare(benchmark::State& state) {
  std::string s1(state.range_x(), '-');
  std::string s2(state.range_x(), '-');
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(s1.compare(s2));
  }
  state.SetComplexityN(state.range_x());
}
BENCHMARK(BM_StringCompare)
    ->RangeMultiplier(2)->Range(1<<10, 1<<18)->Complexity(benchmark::oN);
```

As shown in the following invocation, asymptotic complexity might also be
calculated automatically.

```c++
BENCHMARK(BM_StringCompare)
    ->RangeMultiplier(2)->Range(1<<10, 1<<18)->Complexity();
```

The following code will specify asymptotic complexity with a lambda function,
that might be used to customize high-order term calculation.

```c++
BENCHMARK(BM_StringCompare)->RangeMultiplier(2)
    ->Range(1<<10, 1<<18)->Complexity([](int n)->double{return n; });
```

### Templated benchmarks
Templated benchmarks work the same way: This example produces and consumes
messages of size `sizeof(v)` `range_x` times. It also outputs throughput in the
absence of multiprogramming.

```c++
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
```

Three macros are provided for adding benchmark templates.

```c++
#if __cplusplus >= 201103L // C++11 and greater.
#define BENCHMARK_TEMPLATE(func, ...) // Takes any number of parameters.
#else // C++ < C++11
#define BENCHMARK_TEMPLATE(func, arg1)
#endif
#define BENCHMARK_TEMPLATE1(func, arg1)
#define BENCHMARK_TEMPLATE2(func, arg1, arg2)
```

## Passing arbitrary arguments to a benchmark
In C++11 it is possible to define a benchmark that takes an arbitrary number
of extra arguments. The `BENCHMARK_CAPTURE(func, test_case_name, ...args)`
macro creates a benchmark that invokes `func`  with the `benchmark::State` as
the first argument followed by the specified `args...`.
The `test_case_name` is appended to the name of the benchmark and
should describe the values passed.

```c++
template <class ...ExtraArgs>`
void BM_takes_args(benchmark::State& state, ExtraArgs&&... extra_args) {
  [...]
}
// Registers a benchmark named "BM_takes_args/int_string_test` that passes
// the specified values to `extra_args`.
BENCHMARK_CAPTURE(BM_takes_args, int_string_test, 42, std::string("abc"));
```
Note that elements of `...args` may refer to global variables. Users should
avoid modifying global state inside of a benchmark.

### Multithreaded benchmarks
In a multithreaded test (benchmark invoked by multiple threads simultaneously),
it is guaranteed that none of the threads will start until all have called
`KeepRunning`, and all will have finished before KeepRunning returns false. As
such, any global setup or teardown can be wrapped in a check against the thread
index:

```c++
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
BENCHMARK(BM_MultiThreaded)->Threads(2);
```

If the benchmarked code itself uses threads and you want to compare it to
single-threaded code, you may want to use real-time ("wallclock") measurements
for latency comparisons:

```c++
BENCHMARK(BM_test)->Range(8, 8<<10)->UseRealTime();
```

Without `UseRealTime`, CPU time is used by default.


## Manual timing
For benchmarking something for which neither CPU time nor real-time are
correct or accurate enough, completely manual timing is supported using
the `UseManualTime` function. 

When `UseManualTime` is used, the benchmarked code must call
`SetIterationTime` once per iteration of the `KeepRunning` loop to
report the manually measured time.

An example use case for this is benchmarking GPU execution (e.g. OpenCL
or CUDA kernels, OpenGL or Vulkan or Direct3D draw calls), which cannot
be accurately measured using CPU time or real-time. Instead, they can be
measured accurately using a dedicated API, and these measurement results
can be reported back with `SetIterationTime`.

```c++
static void BM_ManualTiming(benchmark::State& state) {
  int microseconds = state.range_x();
  std::chrono::duration<double, std::micro> sleep_duration {
    static_cast<double>(microseconds)
  };

  while (state.KeepRunning()) {
    auto start = std::chrono::high_resolution_clock::now();
    // Simulate some useful workload with a sleep
    std::this_thread::sleep_for(sleep_duration);
    auto end   = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }
}
BENCHMARK(BM_ManualTiming)->Range(1, 1<<17)->UseManualTime();
```

### Preventing optimisation
To prevent a value or expression from being optimized away by the compiler
the `benchmark::DoNotOptimize(...)` and `benchmark::ClobberMemory()`
functions can be used.

```c++
static void BM_test(benchmark::State& state) {
  while (state.KeepRunning()) {
      int x = 0;
      for (int i=0; i < 64; ++i) {
        benchmark::DoNotOptimize(x += i);
      }
  }
}
```

`DoNotOptimize(<expr>)` forces the  *result* of `<expr>` to be stored in either
memory or a register. For GNU based compilers it acts as read/write barrier
for global memory. More specifically it forces the compiler to flush pending
writes to memory and reload any other values as necessary.

Note that `DoNotOptimize(<expr>)` does not prevent optimizations on `<expr>`
in any way. `<expr>` may even be removed entirely when the result is already
known. For example:

```c++
  /* Example 1: `<expr>` is removed entirely. */
  int foo(int x) { return x + 42; }
  while (...) DoNotOptimize(foo(0)); // Optimized to DoNotOptimize(42);

  /*  Example 2: Result of '<expr>' is only reused */
  int bar(int) __attribute__((const));
  while (...) DoNotOptimize(bar(0)); // Optimized to:
  // int __result__ = bar(0);
  // while (...) DoNotOptimize(__result__);
```

The second tool for preventing optimizations is `ClobberMemory()`. In essence
`ClobberMemory()` forces the compiler to perform all pending writes to global
memory. Memory managed by block scope objects must be "escaped" using
`DoNotOptimize(...)` before it can be clobbered. In the below example
`ClobberMemory()` prevents the call to `v.push_back(42)` from being optimized
away.

```c++
static void BM_vector_push_back(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::vector<int> v;
    v.reserve(1);
    benchmark::DoNotOptimize(v.data()); // Allow v.data() to be clobbered.
    v.push_back(42);
    benchmark::ClobberMemory(); // Force 42 to be written to memory.
  }
}
```

Note that `ClobberMemory()` is only available for GNU based compilers.

### Set time unit manually
If a benchmark runs a few milliseconds it may be hard to visually compare the
measured times, since the output data is given in nanoseconds per default. In
order to manually set the time unit, you can specify it manually:

```c++
BENCHMARK(BM_test)->Unit(benchmark::kMillisecond);
```

## Controlling number of iterations
In all cases, the number of iterations for which the benchmark is run is
governed by the amount of time the benchmark takes. Concretely, the number of
iterations is at least one, not more than 1e9, until CPU time is greater than
the minimum time, or the wallclock time is 5x minimum time. The minimum time is
set as a flag `--benchmark_min_time` or per benchmark by calling `MinTime` on
the registered benchmark object.

## Reporting the mean and standard devation by repeated benchmarks
By default each benchmark is run once and that single result is reported.
However benchmarks are often noisy and a single result may not be representative
of the overall behavior. For this reason it's possible to repeatedly rerun the
benchmark.

The number of runs of each benchmark is specified globally by the
`--benchmark_repetitions` flag or on a per benchmark basis by calling
`Repetitions` on the registered benchmark object. When a benchmark is run
more than once the mean and standard deviation of the runs will be reported.

## Fixtures
Fixture tests are created by
first defining a type that derives from ::benchmark::Fixture and then
creating/registering the tests using the following macros:

* `BENCHMARK_F(ClassName, Method)`
* `BENCHMARK_DEFINE_F(ClassName, Method)`
* `BENCHMARK_REGISTER_F(ClassName, Method)`

For Example:

```c++
class MyFixture : public benchmark::Fixture {};

BENCHMARK_F(MyFixture, FooTest)(benchmark::State& st) {
   while (st.KeepRunning()) {
     ...
  }
}

BENCHMARK_DEFINE_F(MyFixture, BarTest)(benchmark::State& st) {
   while (st.KeepRunning()) {
     ...
  }
}
/* BarTest is NOT registered */
BENCHMARK_REGISTER_F(MyFixture, BarTest)->Threads(2);
/* BarTest is now registered */
```

## Exiting Benchmarks in Error

When errors caused by external influences, such as file I/O and network
communication, occur within a benchmark the
`State::SkipWithError(const char* msg)` function can be used to skip that run
of benchmark and report the error. Note that only future iterations of the
`KeepRunning()` are skipped. Users may explicitly return to exit the
benchmark immediately.

The `SkipWithError(...)` function may be used at any point within the benchmark,
including before and after the `KeepRunning()` loop.

For example:

```c++
static void BM_test(benchmark::State& state) {
  auto resource = GetResource();
  if (!resource.good()) {
      state.SkipWithError("Resource is not good!");
      // KeepRunning() loop will not be entered.
  }
  while (state.KeepRunning()) {
      auto data = resource.read_data();
      if (!resource.good()) {
        state.SkipWithError("Failed to read data!");
        break; // Needed to skip the rest of the iteration.
     }
     do_stuff(data);
  }
}
```

## Output Formats
The library supports multiple output formats. Use the
`--benchmark_format=<tabular|json|csv>` flag to set the format type. `tabular` is
the default format.

The Tabular format is intended to be a human readable format. By default
the format generates color output. Context is output on stderr and the 
tabular data on stdout. Example tabular output looks like:
```
Benchmark                               Time(ns)    CPU(ns) Iterations
----------------------------------------------------------------------
BM_SetInsert/1024/1                        28928      29349      23853  133.097kB/s   33.2742k items/s
BM_SetInsert/1024/8                        32065      32913      21375  949.487kB/s   237.372k items/s
BM_SetInsert/1024/10                       33157      33648      21431  1.13369MB/s   290.225k items/s
```

The JSON format outputs human readable json split into two top level attributes.
The `context` attribute contains information about the run in general, including
information about the CPU and the date.
The `benchmarks` attribute contains a list of ever benchmark run. Example json
output looks like:
``` json
{
  "context": {
    "date": "2015/03/17-18:40:25",
    "num_cpus": 40,
    "mhz_per_cpu": 2801,
    "cpu_scaling_enabled": false,
    "build_type": "debug"
  },
  "benchmarks": [
    {
      "name": "BM_SetInsert/1024/1",
      "iterations": 94877,
      "real_time": 29275,
      "cpu_time": 29836,
      "bytes_per_second": 134066,
      "items_per_second": 33516
    },
    {
      "name": "BM_SetInsert/1024/8",
      "iterations": 21609,
      "real_time": 32317,
      "cpu_time": 32429,
      "bytes_per_second": 986770,
      "items_per_second": 246693
    },
    {
      "name": "BM_SetInsert/1024/10",
      "iterations": 21393,
      "real_time": 32724,
      "cpu_time": 33355,
      "bytes_per_second": 1199226,
      "items_per_second": 299807
    }
  ]
}
```

The CSV format outputs comma-separated values. The `context` is output on stderr
and the CSV itself on stdout. Example CSV output looks like:
```
name,iterations,real_time,cpu_time,bytes_per_second,items_per_second,label
"BM_SetInsert/1024/1",65465,17890.7,8407.45,475768,118942,
"BM_SetInsert/1024/8",116606,18810.1,9766.64,3.27646e+06,819115,
"BM_SetInsert/1024/10",106365,17238.4,8421.53,4.74973e+06,1.18743e+06,
```

## Debug vs Release
By default, benchmark builds as a debug library. You will see a warning in the output when this is the case. To build it as a release library instead, use:

```
cmake -DCMAKE_BUILD_TYPE=Release
```

To enable link-time optimisation, use

```
cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_LTO=true
```

## Linking against the library
When using gcc, it is necessary to link against pthread to avoid runtime exceptions. This is due to how gcc implements std::thread. See [issue #67](https://github.com/google/benchmark/issues/67) for more details.
