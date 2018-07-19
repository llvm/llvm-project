// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: linux, solaris

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(void) {
  struct stat st;
  char *name;

  if (stat("/dev/null", &st))
    exit(1);

  if (!(name = devname(st.st_rdev, S_ISCHR(st.st_mode) ? S_IFCHR : S_IFBLK)))
    exit(1);

  printf("%s\n", name);

  // CHECK: null

  return 0;
}
