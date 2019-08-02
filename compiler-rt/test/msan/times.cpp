// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>


int main(void) {
  struct tms t;
  clock_t res = times(&t);
  assert(res != (clock_t)-1);

  if (t.tms_utime) printf("1\n");
  if (t.tms_stime) printf("2\n");
  if (t.tms_cutime) printf("3\n");
  if (t.tms_cstime) printf("4\n");

  return 0;
}
