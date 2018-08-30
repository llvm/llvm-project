// Test basic realloc functionality.
// RUN: %clang_hwasan %s -o %t
// RUN: %run %t

#include <stdlib.h>
#include <assert.h>

int main() {
  char *x = (char*)realloc(nullptr, 4);
  x[0] = 10;
  x[1] = 20;
  x[2] = 30;
  x[3] = 40;
  char *x1 = (char*)realloc(x, 5);
  assert(x1 != x);  // not necessary true for C,
                    // but true today for hwasan.
  assert(x1[0] == 10 && x1[1] == 20 && x1[2] == 30 && x1[3] == 40);
  x1[4] = 50;

  char *x2 = (char*)realloc(x1, 6);
  x2[5] = 60;
  assert(x2 != x1);
  assert(x2[0] == 10 && x2[1] == 20 && x2[2] == 30 && x2[3] == 40 &&
         x2[4] == 50 && x2[5] == 60);

  char *x3 = (char*)realloc(x2, 6);
  assert(x3 != x2);
  assert(x3[0] == 10 && x3[1] == 20 && x3[2] == 30 && x3[3] == 40 &&
         x3[4] == 50 && x3[5] == 60);

  char *x4 = (char*)realloc(x3, 5);
  assert(x4 != x3);
  assert(x4[0] == 10 && x4[1] == 20 && x4[2] == 30 && x4[3] == 40 &&
         x4[4] == 50);
}
