// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <pthread.h>

#include <sanitizer/msan_interface.h>

int main() {
  cpu_set_t set_x[4];
  int res = pthread_getaffinity_np(pthread_self(), sizeof(set_x), set_x);
  assert(res == 0);
  __msan_check_mem_is_initialized(set_x, sizeof(set_x));

  return 0;
}
