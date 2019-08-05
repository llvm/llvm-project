// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(void) {
  struct stat st;

  assert(!lstat("/dev/null", &st));
  assert(S_ISCHR(st.st_mode));

  return 0;
}
