// RUN: %clangxx -O0 %s -o %t && %run %t

#include <assert.h>
#include <sched.h>
#include <stdio.h>

int main(void) {
  struct sched_param param;
  int res = sched_getparam(0, &param);
  assert(res == 0);
  if (param.sched_priority == 42) printf(".\n");
  return 0;
}
