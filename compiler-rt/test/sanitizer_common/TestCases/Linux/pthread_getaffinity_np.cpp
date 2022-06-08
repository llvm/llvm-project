// RUN: %clangxx -O0 %s -o %t && %run %t

// UNSUPPORTED: android

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/sysinfo.h>

#include <sanitizer/msan_interface.h>

int main() {
  cpu_set_t set_x;
  int res = pthread_getaffinity_np(pthread_self(), sizeof(set_x), &set_x);
  if (res != 0)
    printf("res: %d\n", res);
  assert(res == 0);
  assert(CPU_COUNT_S(sizeof(set_x), &set_x) == get_nprocs());

  return 0;
}
