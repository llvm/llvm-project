// Verify that XRay produces monotonically increasing timestamps on macOS.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

#include "xray/xray_interface.h"
#include <cstdint>
#include <cstdio>

static uint64_t last_ts = 0;
static int monotonic_violations = 0;
static int calls = 0;

#if defined(__x86_64__)
static uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}
#else
// clang-format off
#include <time.h>
// clang-format on
[[clang::xray_never_instrument]] static uint64_t rdtsc() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#endif

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  uint64_t now = rdtsc();
  if (now < last_ts)
    ++monotonic_violations;
  last_ts = now;
  ++calls;
}

[[clang::xray_always_instrument]] void work() {
  volatile int x = 0;
  for (int i = 0; i < 10; ++i)
    x += i;
}

int main() {
  __xray_set_handler(handler);
  __xray_patch();
  for (int i = 0; i < 100; ++i)
    work();
  __xray_unpatch();
  printf("calls=%d violations=%d\n", calls, monotonic_violations);
  return 0;
}

// CHECK: calls={{[0-9]+}} violations=0
