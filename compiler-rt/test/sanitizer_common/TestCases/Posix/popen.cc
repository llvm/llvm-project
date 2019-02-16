// RUN: %clangxx -g %s -o %t && %run %t | FileCheck %s
// CHECK: 1
// CHECK-NEXT: 2

#include <assert.h>
#include <stdio.h>

int main(void) {
  // use a tool that produces different output than input to verify
  // that everything worked correctly
  FILE *fp = popen("sort", "w");
  assert(fp);

  // verify that fileno() returns a meaningful descriptor (needed
  // for the implementation of TSan)
  assert(fileno(fp) != -1);

  assert(fputs("2\n", fp) >= 0);
  assert(fputs("1\n", fp) >= 0);
  assert(pclose(fp) == 0);

  return 0;
}
