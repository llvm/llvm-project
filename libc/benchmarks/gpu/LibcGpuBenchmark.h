#ifndef LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
#define LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H

#include "benchmarks/gpu/BenchmarkLogger.h"
#include "benchmarks/gpu/timing/timing.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/time/clock.h"

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

struct Measurement {
  uint32_t iterations = 0;
  uint64_t elapsed_cycles = 0;
};

class RefinableRuntimeEstimation {
  uint64_t total_cycles = 0;
  uint32_t total_iterations = 0;

public:
  uint64_t update(const Measurement &M) {
    total_cycles += M.elapsed_cycles;
    total_iterations += M.iterations;
    return total_cycles / total_iterations;
  }
};

// Tracks the progression of the runtime estimation
class RuntimeEstimationProgression {
  RefinableRuntimeEstimation rre;

public:
  uint64_t current_estimation = 0;

  double compute_improvement(const Measurement &M) {
    const uint64_t new_estimation = rre.update(M);
    double ratio =
        (static_cast<double>(current_estimation) / new_estimation) - 1.0;

    // Get absolute value
    if (ratio < 0)
      ratio *= -1;

    current_estimation = new_estimation;
    return ratio;
  }
};

struct BenchmarkResult {
  uint64_t cycles = 0;
  double standard_deviation = 0;
  uint64_t min = UINT64_MAX;
  uint64_t max = 0;
  uint32_t samples = 0;
  uint32_t total_iterations = 0;
  clock_t total_time = 0;
};

BenchmarkResult
benchmark(const BenchmarkOptions &options,
          const cpp::function<uint64_t(uint32_t)> &wrapper_func);

class Benchmark {
  const cpp::function<uint64_t(uint32_t)> func;
  const cpp::string_view suite_name;
  const cpp::string_view test_name;
  const uint32_t num_threads;

public:
  Benchmark(cpp::function<uint64_t(uint32_t)> func, char const *suite_name,
            char const *test_name, uint32_t num_threads)
      : func(func), suite_name(suite_name), test_name(test_name),
        num_threads(num_threads) {
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
    return benchmark(options, func);
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
