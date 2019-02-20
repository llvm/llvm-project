// RUN: %clang_esan_wset -O0 %s -o %t 2>&1
// RUN: %run %t 2>&1 | FileCheck %s

// FIXME: Re-enable once PR33590 is fixed.
// UNSUPPORTED: x86_64
// Stucks at init and no clone feature equivalent.
// UNSUPPORTED: freebsd

#include <sanitizer/esan_interface.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

const int size = 0x1 << 25; // 523288 cache lines

int main(int argc, char **argv) {
  char *buf = (char *)mmap(0, size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  // To avoid flakiness stemming from whether the sideline thread
  // is scheduled enough on a loaded test machine, we coordinate
  // with esan itself:
  if (__esan_get_sample_count) {
    while (__esan_get_sample_count() < 4) {
      for (int i = 0; i < size; ++i)
        buf[i] = i;
      sched_yield();
    }
  }
  munmap(buf, size);
  // We only check for a few samples here to reduce the chance of flakiness.
  // CHECK:      =={{[0-9]+}}== Total number of samples: {{[0-9]+}}
  // CHECK-NEXT: =={{[0-9]+}}== Samples array #0 at period 20 ms
  // CHECK-NEXT: =={{[0-9]+}}==#   0: {{[ 0-9]+}} {{KB|MB|Bytes}} ({{[ 0-9]+}} cache lines)
  // CHECK-NEXT: =={{[0-9]+}}==#   1: {{[ 0-9]+}} {{KB|MB|Bytes}} ({{[ 0-9]+}} cache lines)
  // CHECK-NEXT: =={{[0-9]+}}==#   2: {{[ 0-9]+}} {{KB|MB|Bytes}} ({{[ 0-9]+}} cache lines)
  // CHECK-NEXT: =={{[0-9]+}}==#   3: {{[ 0-9]+}} {{KB|MB|Bytes}} ({{[ 0-9]+}} cache lines)
  // CHECK:      =={{[0-9]+}}== Samples array #1 at period 80 ms
  // CHECK-NEXT: =={{[0-9]+}}==#   0: {{[ 0-9]+}} {{KB|MB|Bytes}} ({{[ 0-9]+}} cache lines)
  // CHECK:      =={{[0-9]+}}== Samples array #2 at period 320 ms
  // CHECK:      =={{[0-9]+}}== Samples array #3 at period 1280 ms
  // CHECK:      =={{[0-9]+}}== Samples array #4 at period 5120 ms
  // CHECK:      =={{[0-9]+}}== Samples array #5 at period 20 sec
  // CHECK:      =={{[0-9]+}}== Samples array #6 at period 81 sec
  // CHECK:      =={{[0-9]+}}== Samples array #7 at period 327 sec
  // CHECK: {{.*}} EfficiencySanitizer: the total working set size: 32 MB (5242{{[0-9][0-9]}} cache lines)
  return 0;
}
