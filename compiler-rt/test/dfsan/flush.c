// Tests dfsan_flush().
// RUN: %clang_dfsan %s -o %t && %run %t
#include <sanitizer/dfsan_interface.h>
#include <assert.h>
#include <stdlib.h>

int global;

int main() {
  int local;
  int *heap = (int*)malloc(sizeof(int));

  dfsan_set_label(10, &global, sizeof(global));
  dfsan_set_label(20, &local, sizeof(local));
  dfsan_set_label(30, heap, sizeof(*heap));

  assert(dfsan_get_label(global) == 10);
  assert(dfsan_get_label(local) == 20);
  assert(dfsan_get_label(*heap) == 30);

  dfsan_flush();

  assert(dfsan_get_label(global) == 0);
  assert(dfsan_get_label(local) == 0);
  assert(dfsan_get_label(*heap) == 0);

  free(heap);
}
