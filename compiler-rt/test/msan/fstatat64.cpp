// REQUIRES: linux
// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <cassert>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>

int main(void) {
  struct stat64 st;
  int dirfd = open("/dev", O_RDONLY);
  if (fstatat64(dirfd, "null", &st, 0))
    exit(1);

  assert(S_ISCHR(st.st_mode));

  return 0;
}
