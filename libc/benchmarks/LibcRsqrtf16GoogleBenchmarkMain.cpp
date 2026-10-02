#include "benchmark/benchmark.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/rsqrtf16.h"

#include <stddef.h>
#include <stdint.h>

namespace {

using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

constexpr uint16_t INPUTS[] = {
    // Subnormals.
    0x0001,
    0x0002,
    0x0004,
    0x0008,
    0x0010,
    0x0020,
    0x0040,
    0x0080,
    0x0100,
    0x0200,
    0x03ff,
    // Normals spread across all exponent ranges.
    0x0400,
    0x0401,
    0x047f,
    0x0555,
    0x07ff,
    0x0800,
    0x0c00,
    0x1000,
    0x1400,
    0x1800,
    0x1c00,
    0x2000,
    0x2400,
    0x2800,
    0x2c00,
    0x3000,
    0x3400,
    0x3800,
    0x3c00,
    0x3c01,
    0x3d55,
    0x3fff,
    0x4000,
    0x4400,
    0x4800,
    0x4c00,
    0x5000,
    0x5400,
    0x5800,
    0x5c00,
    0x6000,
    0x6400,
    0x6800,
    0x6c00,
    0x7000,
    0x7400,
    0x7800,
    0x7bff,
};

constexpr size_t INPUT_COUNT = sizeof(INPUTS) / sizeof(INPUTS[0]);

float16 get_input(uint16_t bits) { return FPBits(bits).get_val(); }

float16 rsqrtf16_integer_finite(float16 x) {
  FPBits xbits(x);
  return LIBC_NAMESPACE::math::rsqrtf16_internal::rsqrtf16_no_float(
      xbits.uintval() & 0x7fff);
}

void BM_Rsqrtf16HostFpu(benchmark::State &state) {
  for (auto _ : state) {
    for (uint16_t bits : INPUTS)
      benchmark::DoNotOptimize(LIBC_NAMESPACE::math::rsqrtf16(get_input(bits)));
  }
  state.SetItemsProcessed(state.iterations() * INPUT_COUNT);
}

void BM_Rsqrtf16IntegerFallbackFiniteWrapper(benchmark::State &state) {
  for (auto _ : state) {
    for (uint16_t bits : INPUTS)
      benchmark::DoNotOptimize(rsqrtf16_integer_finite(get_input(bits)));
  }
  state.SetItemsProcessed(state.iterations() * INPUT_COUNT);
}

void BM_Rsqrtf16IntegerFallback(benchmark::State &state) {
  for (auto _ : state) {
    for (uint16_t bits : INPUTS)
      benchmark::DoNotOptimize(
          LIBC_NAMESPACE::math::rsqrtf16_internal::rsqrtf16_no_float(bits));
  }
  state.SetItemsProcessed(state.iterations() * INPUT_COUNT);
}

} // namespace

BENCHMARK(BM_Rsqrtf16HostFpu);
BENCHMARK(BM_Rsqrtf16IntegerFallbackFiniteWrapper);
BENCHMARK(BM_Rsqrtf16IntegerFallback);
