// RUN: %clang %s -Wl,-as-needed -o %t && %run %t
#include <time.h>
#include <unistd.h>
#include <assert.h>

long cpu_ns() {
  clockid_t clk;
  struct timespec ts;
  int res = clock_getcpuclockid(getpid(), &clk);
  assert(!res);
  res = clock_gettime(clk, &ts);
  assert(!res);
  return ts.tv_nsec;
}

int main() {
  long cpuns = cpu_ns();
  asm volatile ("" :: "r"(cpuns));
  return 0;
}
