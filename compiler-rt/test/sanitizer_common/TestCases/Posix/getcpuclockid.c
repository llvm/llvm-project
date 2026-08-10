// RUN: %clang -pthread %s -Wl,-as-needed -o %t && %run %t
//
// UNSUPPORTED: darwin, target={{.*solaris.*}}

#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>

long cpu_ns() {
  clockid_t clk;
  struct timespec ts;
  int res = clock_getcpuclockid(getpid(), &clk);
  assert(!res);
  res = clock_gettime(clk, &ts);
  assert(!res);
  return ts.tv_nsec;
}

long th_cpu_ns() {
  clockid_t clk;
  struct timespec ts;
  int res = pthread_getcpuclockid(pthread_self(), &clk);
  assert(!res);
  res = clock_gettime(clk, &ts);
  assert(!res);
  return ts.tv_nsec;
}

int main() {
  long cpuns = cpu_ns();
  asm volatile ("" :: "r"(cpuns));
  cpuns = th_cpu_ns();
  asm volatile ("" :: "r"(cpuns));
  return 0;
}
