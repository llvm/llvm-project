// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t | FileCheck %s

#include <windows.h>
#include <stdio.h>

int main(void) {
  static const char *foo = "foobarspam";
  printf("Global string is `%s`\n", foo);
// CHECK: Global string is `foobarspam`
  return 0;
}
