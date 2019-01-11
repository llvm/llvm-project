// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
  printf("getusershell\n");

  setusershell();
  char *fentry = getusershell();

  printf("First entry: '%s'\n", fentry);

  endusershell();

  return 0;
  // CHECK: getusershell
  // CHECK: First entry: '{{.*}}'
}
