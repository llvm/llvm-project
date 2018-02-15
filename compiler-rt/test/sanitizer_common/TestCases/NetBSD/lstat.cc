// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <stdlib.h>
#include <sys/stat.h>

int main(void) {
  struct stat st;

  if (lstat("/dev/null", &st))
    exit(1);

  if (!S_ISCHR(st.st_mode))
    exit(1);

  return 0;
}
