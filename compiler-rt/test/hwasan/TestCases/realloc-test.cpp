// Test basic realloc functionality.
// RUN: %clang_hwasan %s -o %t && %run %t
// RUN: %clang_hwasan %s -DREALLOCARRAY -o %t && %run %t

#include <assert.h>
#include <sanitizer/hwasan_interface.h>

#ifdef REALLOCARRAY
extern "C" void *reallocarray(void *, size_t nmemb, size_t size);
#define REALLOC(p, s) reallocarray(p, 1, s)
#else
#include <stdlib.h>
#define REALLOC(p, s) realloc(p, s)
#endif

int main() {
  __hwasan_enable_allocator_tagging();
  char *x = (char*)REALLOC(nullptr, 4);
  x[0] = 10;
  x[1] = 20;
  x[2] = 30;
  x[3] = 40;
  char *x1 = (char*)REALLOC(x, 5);
  assert(x1 != x);  // not necessary true for C,
                    // but true today for hwasan.
  assert(x1[0] == 10 && x1[1] == 20 && x1[2] == 30 && x1[3] == 40);
  x1[4] = 50;

  char *x2 = (char*)REALLOC(x1, 6);
  x2[5] = 60;
  assert(x2 != x1);
  assert(x2[0] == 10 && x2[1] == 20 && x2[2] == 30 && x2[3] == 40 &&
         x2[4] == 50 && x2[5] == 60);

  char *x3 = (char*)REALLOC(x2, 6);
  assert(x3 != x2);
  assert(x3[0] == 10 && x3[1] == 20 && x3[2] == 30 && x3[3] == 40 &&
         x3[4] == 50 && x3[5] == 60);

  char *x4 = (char*)REALLOC(x3, 5);
  assert(x4 != x3);
  assert(x4[0] == 10 && x4[1] == 20 && x4[2] == 30 && x4[3] == 40 &&
         x4[4] == 50);
}
