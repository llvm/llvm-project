// RUN: %clangxx -O0 %s -o %t && %run %t

// Android does not implement pthread_getaffinity_np.
// (Note: libresolv is integrated with libc, but apparently only
// sched_getaffinity).
// UNSUPPORTED: android

#include <assert.h>
#include <pthread.h>
#include <sys/sysinfo.h>

#include <sanitizer/msan_interface.h>

int main() {
  cpu_set_t set_x[4];
  pthread_t tid = pthread_self();
  int res = pthread_getaffinity_np(tid, sizeof(set_x), set_x);
  assert(res == 0);
  assert(CPU_COUNT_S(sizeof(set_x), set_x) == get_nprocs());

  return 0;
}
