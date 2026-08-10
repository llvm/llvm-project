// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: target={{.*(linux|solaris).*}}

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(void) {
  struct stat st;
  char *name;

  assert(!stat("/dev/null", &st));
  assert((name = devname(st.st_rdev, S_ISCHR(st.st_mode) ? S_IFCHR : S_IFBLK)));

  printf("%s\n", name);

  // CHECK: null

  return 0;
}
