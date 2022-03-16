// REQUIRES: linux
// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <stdlib.h>
#include <sys/stat.h>

int main(void) {
  struct stat64 st;
  if (fstat64(0, &st))
    exit(1);

  if (S_ISBLK(st.st_mode))
    exit(0);

  return 0;
}
