// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: linux, darwin, solaris

#include <langinfo.h>

#include <stdio.h>

int main(void) {
  printf("nl_langinfo\n");

  char *info = nl_langinfo(DAY_1);

  printf("DAY_1='%s'\n", info);

  // CHECK: nl_langinfo
  // CHECK: DAY_1='{{.*}}'

  return 0;
}
