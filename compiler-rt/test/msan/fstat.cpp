// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <sys/stat.h>
#include <stdlib.h>

int main(void) {
  struct stat st;
  if (fstat(0, &st))
    exit(1);

  if (S_ISBLK(st.st_mode))
    exit(0);

  return 0;
}
