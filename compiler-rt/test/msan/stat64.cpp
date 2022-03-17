// REQUIRES: linux
// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <cassert>
#include <cstdlib>
#include <sys/stat.h>

int main(void) {
  struct stat64 st;
  if (stat64("/dev/null", &st))
    exit(1);

  assert(S_ISCHR(st.st_mode));

  return 0;
}
