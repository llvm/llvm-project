// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: linux, solaris

#include <sys/cdefs.h>
#include <sys/stat.h>

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  struct stat st;
  char name[10];
  mode_t type;

  if (stat("/dev/null", &st))
    exit(1);

  type = S_ISCHR(st.st_mode) ? S_IFCHR : S_IFBLK;

  if (devname_r(st.st_rdev, type, name, sizeof(name)))
    exit(1);

  printf("%s\n", name);

  // CHECK: null

  return 0;
}
