#ifndef LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
#define LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H

#include "benchmarks/gpu/timing/timing.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace benchmarks {

struct BenchmarkOptions {
  uint32_t initial_iterations = 1;
  uint32_t min_iterations = 1;
  uint32_t max_iterations = 10000000;
  uint32_t min_samples = 4;
  uint32_t max_samples = 1000;
  int64_t min_duration = 500 * 1000;         // 500 * 1000 nanoseconds = 500 us
  int64_t max_duration = 1000 * 1000 * 1000; // 1e9 nanoseconds = 1 second
  double epsilon = 0.0001;
  double scaling_factor = 1.4;
};

class RefinableRuntimeEstimator {
  uint32_t iterations = 0;
  uint64_t sum_of_cycles = 0;
  uint64_t sum_of_squared_cycles = 0;

public:
  void update(uint64_t cycles) noexcept {
    iterations += 1;
    sum_of_cycles += cycles;
    sum_of_squared_cycles += cycles * cycles;
  }

  void update(const RefinableRuntimeEstimator &other) noexcept {
    iterations += other.iterations;
    sum_of_cycles += other.sum_of_cycles;
    sum_of_squared_cycles += other.sum_of_squared_cycles;
  }

  double get_mean() const noexcept {
    if (iterations == 0)
      return 0.0;

    return static_cast<double>(sum_of_cycles) / iterations;
  }

  double get_variance() const noexcept {
    if (iterations == 0)
      return 0.0;

    const double num = static_cast<double>(iterations);
    const double sum_x = static_cast<double>(sum_of_cycles);
    const double sum_x2 = static_cast<double>(sum_of_squared_cycles);

    const double mean_of_squares = sum_x2 / num;
    const double mean = sum_x / num;
    const double mean_squared = mean * mean;
    const double variance = mean_of_squares - mean_squared;

    return variance < 0.0 ? 0.0 : variance;
  }

  double get_stddev() const noexcept {
    return fputil::sqrt<double>(get_variance());
  }

  uint32_t get_iterations() const noexcept { return iterations; }
};

// Tracks the progression of the runtime estimation
class RuntimeEstimationProgression {
  RefinableRuntimeEstimator estimator;
  double current_mean = 0.0;

public:
  const RefinableRuntimeEstimator &get_estimator() const noexcept {
    return estimator;
  }

  double
  compute_improvement(const RefinableRuntimeEstimator &sample_estimator) {
    if (sample_estimator.get_iterations() == 0)
      return 1.0;

    estimator.update(sample_estimator);

    const double new_mean = estimator.get_mean();
    if (current_mean == 0.0 || new_mean == 0.0) {
      current_mean = new_mean;
      return 1.0;
    }

    double ratio = (current_mean / new_mean) - 1.0;
    if (ratio < 0)
      ratio = -ratio;

    current_mean = new_mean;
    return ratio;
  }
};

struct BenchmarkResult {
  uint64_t total_iterations = 0;
  double cycles = 0;
  double standard_deviation = 0;
  uint64_t min = UINT64_MAX;
  uint64_t max = 0;
};

struct BenchmarkTarget {
  using IndexedFnPtr = uint64_t (*)(uint32_t);
  using IndexlessFnPtr = uint64_t (*)();

  enum class Kind : uint8_t { Indexed, Indexless } kind;
  union {
    IndexedFnPtr indexed_fn_ptr;
    IndexlessFnPtr indexless_fn_ptr;
  };

  LIBC_INLINE BenchmarkTarget(IndexedFnPtr func)
      : kind(Kind::Indexed), indexed_fn_ptr(func) {}
  LIBC_INLINE BenchmarkTarget(IndexlessFnPtr func)
      : kind(Kind::Indexless), indexless_fn_ptr(func) {}

  LIBC_INLINE uint64_t operator()([[maybe_unused]] uint32_t call_index) const {
    return kind == Kind::Indexed ? indexed_fn_ptr(call_index)
                                 : indexless_fn_ptr();
  }
};

BenchmarkResult benchmark(const BenchmarkOptions &options,
                          const BenchmarkTarget &target);

class Benchmark {
  const BenchmarkTarget target;
  const cpp::string_view suite_name;
  const cpp::string_view test_name;
  const uint32_t num_threads;

public:
  Benchmark(uint64_t (*f)(), const char *suite, const char *test,
            uint32_t threads)
      : target(BenchmarkTarget(f)), suite_name(suite), test_name(test),
        num_threads(threads) {
    add_benchmark(this);
  }

  Benchmark(uint64_t (*f)(uint32_t), char const *suite_name,
            char const *test_name, uint32_t num_threads)
      : target(BenchmarkTarget(f)), suite_name(suite_name),
        test_name(test_name), num_threads(num_threads) {
    add_benchmark(this);
  }

  static void run_benchmarks();
  const cpp::string_view get_suite_name() const { return suite_name; }
  const cpp::string_view get_test_name() const { return test_name; }

protected:
  static void add_benchmark(Benchmark *benchmark);

private:
  BenchmarkResult run() {
    BenchmarkOptions options;
    return benchmark(options, target);
  }
};

class RandomGenerator {
  uint64_t state;

  static LIBC_INLINE uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = (x ^ (x >> 31));
    return x ? x : 0x9E3779B97F4A7C15ULL;
  }

public:
  explicit LIBC_INLINE RandomGenerator(uint64_t seed) noexcept
      : state(splitmix64(seed)) {}

  LIBC_INLINE uint64_t next64() noexcept {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 0x2545F4914F6CDD1DULL;
  }

  LIBC_INLINE uint32_t next32() noexcept {
    return static_cast<uint32_t>(next64() >> 32);
  }
};

// We want random floating-point values whose *unbiased* exponent e is
// approximately uniform in [min_exp, max_exp]. That is,
//   2^min_exp <= |value| < 2^(max_exp + 1).
// Caveats / boundaries:
// - e = -EXP_BIAS  ==> subnormal range (biased exponent = 0). We ensure a
//                      non-zero mantissa so we don't accidentally produce 0.
// - e in [1 - EXP_BIAS, EXP_BIAS] ==> normal numbers.
// - e = EXP_BIAS + 1 ==> Inf/NaN. We do not include it by default; max_exp
//                        defaults to EXP_BIAS.
template <typename T>
static T
get_rand_input(RandomGenerator &rng,
               int min_exp = -LIBC_NAMESPACE::fputil::FPBits<T>::EXP_BIAS,
               int max_exp = LIBC_NAMESPACE::fputil::FPBits<T>::EXP_BIAS) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using Storage = typename FPBits::StorageType;

  // Sanitize and clamp requested range to what the format supports
  if (min_exp > max_exp) {
    auto tmp = min_exp;
    min_exp = max_exp;
    max_exp = tmp;
  };
  min_exp = cpp::max(min_exp, -FPBits::EXP_BIAS);
  max_exp = cpp::min(max_exp, FPBits::EXP_BIAS);

  // Sample unbiased exponent e uniformly in [min_exp, max_exp] without modulo
  // bias
  auto sample_in_range = [&](uint64_t r) -> int32_t {
    const uint64_t range = static_cast<uint64_t>(
        static_cast<int64_t>(max_exp) - static_cast<int64_t>(min_exp) + 1);
    const uint64_t threshold = (-range) % range;
    while (r < threshold)
      r = rng.next64();
    return static_cast<int32_t>(min_exp + static_cast<int64_t>(r % range));
  };
  const int32_t e = sample_in_range(rng.next64());

  // Start from random bits to get random sign and mantissa
  FPBits xbits([&] {
    if constexpr (cpp::is_same_v<T, double>)
      return FPBits(rng.next64());
    else
      return FPBits(rng.next32());
  }());

  if (e == -FPBits::EXP_BIAS) {
    // Subnormal: biased exponent must be 0; ensure mantissa != 0 to avoid 0
    xbits.set_biased_exponent(Storage(0));
    if (xbits.get_mantissa() == Storage(0))
      xbits.set_mantissa(Storage(1));
  } else {
    // Normal: biased exponent in [1, 2 * FPBits::EXP_BIAS]
    const int32_t biased = e + FPBits::EXP_BIAS;
    xbits.set_biased_exponent(static_cast<Storage>(biased));
  }
  return xbits.get_val();
}

template <typename T> class MathPerf {
  static LIBC_INLINE uint64_t make_seed(uint64_t base_seed, uint64_t salt) {
    const uint64_t tid = gpu::get_thread_id();
    return base_seed ^ (salt << 32) ^ (tid * 0x9E3779B97F4A7C15ULL);
  }

public:
  // Returns cycles-per-call (lower is better)
  template <size_t N = 1>
  static uint64_t run_throughput_in_range(T f(T), int min_exp, int max_exp,
                                          uint32_t call_index) {
    cpp::array<T, N> inputs;

    uint64_t base_seed = static_cast<uint64_t>(call_index);
    uint64_t salt = static_cast<uint64_t>(N);
    RandomGenerator rng(make_seed(base_seed, salt));

    for (size_t i = 0; i < N; ++i)
      inputs[i] = get_rand_input<T>(rng, min_exp, max_exp);

    uint64_t total_time = LIBC_NAMESPACE::throughput(f, inputs);

    return total_time / N;
  }

  // Returns cycles-per-call (lower is better)
  template <size_t N = 1>
  static uint64_t run_throughput_in_range(T f(T, T), int arg1_min_exp,
                                          int arg1_max_exp, int arg2_min_exp,
                                          int arg2_max_exp,
                                          uint32_t call_index) {
    cpp::array<T, N> inputs1;
    cpp::array<T, N> inputs2;

    uint64_t base_seed = static_cast<uint64_t>(call_index);
    uint64_t salt = static_cast<uint64_t>(N);
    RandomGenerator rng(make_seed(base_seed, salt));

    for (size_t i = 0; i < N; ++i) {
      inputs1[i] = get_rand_input<T>(rng, arg1_min_exp, arg1_max_exp);
      inputs2[i] = get_rand_input<T>(rng, arg2_min_exp, arg2_max_exp);
    }

    uint64_t total_time = LIBC_NAMESPACE::throughput(f, inputs1, inputs2);

    return total_time / N;
  }
};

} // namespace benchmarks
} // namespace LIBC_NAMESPACE_DECL

// Passing -1 indicates the benchmark should be run with as many threads as
// allocated by the user in the benchmark's CMake.
#define BENCHMARK(SuiteName, TestName, Func)                                   \
  LIBC_NAMESPACE::benchmarks::Benchmark SuiteName##_##TestName##_Instance(     \
      Func, #SuiteName, #TestName, -1)

#define BENCHMARK_N_THREADS(SuiteName, TestName, Func, NumThreads)             \
  LIBC_NAMESPACE::benchmarks::Benchmark SuiteName##_##TestName##_Instance(     \
      Func, #SuiteName, #TestName, NumThreads)

#define SINGLE_THREADED_BENCHMARK(SuiteName, TestName, Func)                   \
  BENCHMARK_N_THREADS(SuiteName, TestName, Func, 1)

#define SINGLE_WAVE_BENCHMARK(SuiteName, TestName, Func)                       \
  BENCHMARK_N_THREADS(SuiteName, TestName, Func,                               \
                      LIBC_NAMESPACE::gpu::get_lane_size())

#endif // LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
