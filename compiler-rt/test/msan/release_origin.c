// RUN: %clang_msan -fsanitize-memory-track-origins=0 -O0 %s -o %t && env MSAN_OPTIONS=soft_rss_limit_mb=60:verbosity=1:allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s -implicit-check-not="soft rss limit" -check-prefixes=CHECK,NOORIG
// RUN: %clang_msan -fsanitize-memory-track-origins=2 -O0 %s -o %t && env MSAN_OPTIONS=soft_rss_limit_mb=150:verbosity=1:allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s -implicit-check-not="soft rss limit" -check-prefixes=CHECK,ORIGIN

#define _GNU_SOURCE
#include <assert.h>
#ifdef __linux__
#  include <sched.h>
#endif
#include <sanitizer/allocator_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void *p;

int main(int argc, char **argv) {
#ifdef __linux__
  int cpu = sched_getcpu();
  if (cpu >= 0 && cpu < CPU_SETSIZE) {
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu, &cpus);
    sched_setaffinity(0, sizeof(cpus), &cpus);
  }
#endif
  int s = 100 * 1024 * 1024;
  fprintf(stderr, "malloc\n");
  p = malloc(s);
  sleep(1);
  fprintf(stderr, "memset\n");
  memset(p, 1, s);
  sleep(1);
  fprintf(stderr, "free\n");
  free(p);
  sleep(1);
  return 0;
}

// CHECK-LABEL: malloc

// Non origin mode allocate ~100M for shadow.
// Origin mode allocate ~100M for shadow and ~100M for origin.
// CHECK: soft rss limit exhausted

// CHECK-LABEL: memset

// Memset reserve modified pages, frees ~100M for shadow. So not change in RSS for non-origin mode.
// Origin mode also frees ~100M of origins, so 'unexhausted'.
// ORIGIN: soft rss limit unexhausted

// CHECK-LABEL: free

// Now non-origin release all everything.
// NOORIG: soft rss limit unexhausted
