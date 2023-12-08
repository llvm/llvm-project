// REQUIRES: linux
// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(void) {
  struct stat64 st;

  assert(!lstat64("/dev/null", &st));
#if defined(__sun__) && defined(__svr4__)
  assert(S_ISLNK(st.st_mode));
#else
  assert(S_ISCHR(st.st_mode));
#endif

  return 0;
}
