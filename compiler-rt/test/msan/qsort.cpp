// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -DPOISON -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <errno.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sanitizer/msan_interface.h>

constexpr size_t kSize1 = 27;
constexpr size_t kSize2 = 7;

bool seen2;

void dummy(long a, long b, long c, long d, long e) {}

void poison_stack_and_param() {
  char x[10000];
  int y;
  dummy(y, y, y, y, y);
}

__attribute__((always_inline)) int cmp(long a, long b) {
  if (a < b)
    return -1;
  else if (a > b)
    return 1;
  else
    return 0;
}

int compar2(const void *a, const void *b) {
  assert(a);
  assert(b);
  __msan_check_mem_is_initialized(a, sizeof(long));
  __msan_check_mem_is_initialized(b, sizeof(long));
  seen2 = true;
  poison_stack_and_param();
  return cmp(*(long *)a, *(long *)b);
}

int compar1(const void *a, const void *b) {
  assert(a);
  assert(b);
  __msan_check_mem_is_initialized(a, sizeof(long));
  __msan_check_mem_is_initialized(b, sizeof(long));

  long *p = new long[kSize2];
  // kind of random
  for (int i = 0; i < kSize2; ++i)
    p[i] = i * 2 + (i % 3 - 1) * 3;
  qsort(p, kSize1, sizeof(long), compar2);
  __msan_check_mem_is_initialized(p, sizeof(long) * kSize2);
  delete[] p;

  poison_stack_and_param();
  return cmp(*(long *)a, *(long *)b);
}

int main(int argc, char *argv[]) {
  long *p = new long[kSize1];
  // kind of random
  for (int i = 0; i < kSize1; ++i)
    p[i] = i * 2 + (i % 3 - 1) * 3;
  poison_stack_and_param();
#ifdef POISON
  __msan_poison(p + 1, sizeof(long));
  // CHECK: Uninitialized bytes in __msan_check_mem_is_initialized at offset 0 inside [{{.*}}, 8)
#endif
  qsort(p, kSize1, sizeof(long), compar1);
  __msan_check_mem_is_initialized(p, sizeof(long) * kSize1);
  assert(seen2);
  delete[] p;

  p = new long[0];
  qsort(p, 0, sizeof(long), compar1);
  delete[] p;

  qsort(nullptr, 0, sizeof(long), compar1);

  return 0;
}
