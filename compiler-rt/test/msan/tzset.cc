// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// XFAIL: freebsd

#include <stdlib.h>
#include <string.h>
#include <time.h>

extern char *tzname[2];

int main(void) {
  if (!strlen(tzname[0]) || !strlen(tzname[1]))
    exit(1);
  tzset();
  if (!strlen(tzname[0]) || !strlen(tzname[1]))
    exit(1);
  return 0;
}
