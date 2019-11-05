// On Darwin, the man page states that "both fputs() and puts() print `(null)'
// if str is NULL."
//
// RUN: %clangxx -g %s -o %t && %run %t | FileCheck %s
// CHECK: {{^\(null\)---\(null\)$}}

#include <assert.h>
#include <stdio.h>

int main(void) {
  assert(fputs(NULL, stdout) >= 0);
  fputs("---", stdout);
  assert(puts(NULL) >= 0);

  return 0;
}
