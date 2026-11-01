// NOTE: Do not autogenerate. Check runtime counts and application output.
// Exercise the installed GPU profiling helper with zero or one from each
// active lane. Also check partial/sparse masks and both counter channels
// across multiple waves. Uniform counters record contributions
// made with a full active mask; they do not classify the step's uniformity.
// REQUIRES: hip, amdgpu
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O2 \
// RUN:   -fprofile-instr-generate %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/profile.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.dir/a.out | FileCheck %s
// CHECK: cases=11 mismatches=0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

extern "C" __device__ void __llvm_profile_instrument_gpu(uint64_t *, uint64_t *,
                                                         uint64_t);

__global__ void accumulate(uint64_t *counts, const uint64_t *steps,
                           const unsigned char *active) {
  unsigned i = threadIdx.x;
  if (active[i])
    __llvm_profile_instrument_gpu(counts, counts + 1, steps[i]);
}

static void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "HIP operation returned %s\n",
                 hipGetErrorString(error));
    std::exit(2);
  }
}

static uint64_t stepValue(unsigned kind, unsigned lane) {
  switch (kind) {
  case 0:
    return 0;
  case 1:
    return 1;
  case 2:
    return lane % 2;
  default:
    return 1 - lane % 2;
  }
}

static bool isActive(unsigned test, unsigned wave, unsigned lane,
                     unsigned width) {
  if (test < 4)
    return true;
  if (test < 6)
    return lane >= 3 && lane < width - 5;
  if (test < 8)
    return lane % 3 != 0;
  if (test == 8)
    return lane == width - 1;
  if (test == 9)
    return false;
  return wave == 0 || (wave == 2 && lane % 2 != 0);
}

int main() {
  hipDeviceProp_t properties;
  check(hipGetDeviceProperties(&properties, 0));
  const unsigned width = properties.warpSize;
  const unsigned n = 4 * width;
  if (n > 256)
    return 2;
  uint64_t *counts, *steps;
  unsigned char *active;
  check(hipMalloc(&counts, 2 * sizeof(uint64_t)));
  check(hipMalloc(&steps, n * sizeof(uint64_t)));
  check(hipMalloc(&active, n));
  uint64_t values[256];
  unsigned char mask[256];
  unsigned errors = 0;
  for (unsigned test = 0; test < 11; ++test) {
    uint64_t expected[2] = {};
    for (unsigned wave = 0; wave < 4; ++wave) {
      uint64_t sum = 0;
      unsigned population = 0;
      for (unsigned lane = 0; lane < width; ++lane) {
        unsigned i = wave * width + lane;
        unsigned kind = test < 4 ? test : (test % 2 == 0 ? 2 : 1);
        uint64_t value = stepValue(kind, lane);
        bool enabled = isActive(test, wave, lane, width);
        values[i] = value;
        mask[i] = enabled;
        if (enabled) {
          sum += value;
          ++population;
        }
      }
      expected[0] += sum;
      if (population == width)
        expected[1] += sum;
    }
    check(
        hipMemcpy(steps, values, n * sizeof(uint64_t), hipMemcpyHostToDevice));
    check(hipMemcpy(active, mask, n, hipMemcpyHostToDevice));
    check(hipMemset(counts, 0, 2 * sizeof(uint64_t)));
    accumulate<<<1, n>>>(counts, steps, active);
    check(hipGetLastError());
    uint64_t actual[2];
    check(hipMemcpy(actual, counts, sizeof(actual), hipMemcpyDeviceToHost));
    for (unsigned channel = 0; channel < 2; ++channel) {
      if (actual[channel] != expected[channel]) {
        ++errors;
        std::printf("case=%u channel=%u actual=%llu expected=%llu\n", test,
                    channel, (unsigned long long)actual[channel],
                    (unsigned long long)expected[channel]);
      }
    }
  }
  check(hipFree(active));
  check(hipFree(steps));
  check(hipFree(counts));
  std::printf("cases=11 mismatches=%u\n", errors);
  return errors != 0;
}
