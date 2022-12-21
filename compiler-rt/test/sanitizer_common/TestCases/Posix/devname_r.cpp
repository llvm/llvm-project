// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: target={{.*(linux|solaris).*}}

#include <sys/cdefs.h>
#include <sys/stat.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  struct stat st;
  char name[100];
  mode_t type;

  assert(!stat("/dev/null", &st));

  type = S_ISCHR(st.st_mode) ? S_IFCHR : S_IFBLK;

#if defined(__NetBSD__)
  assert(!devname_r(st.st_rdev, type, name, sizeof(name)));
#else
  assert(devname_r(st.st_rdev, type, name, sizeof(name)));
#endif

  printf("%s\n", name);

  // CHECK: null

  return 0;
}
