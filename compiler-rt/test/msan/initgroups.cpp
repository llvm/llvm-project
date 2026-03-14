// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <sys/types.h>
#include <grp.h>
#include <unistd.h>  // FreeBSD declares initgroups() here.

int main(void) {
  initgroups("root", 0);
  // The above fails unless you are root. Does not matter, MSan false positive
  // (which we are testing for) happens anyway.
  return 0;
}
