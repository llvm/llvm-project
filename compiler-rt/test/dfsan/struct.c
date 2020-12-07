// RUN: %clang_dfsan %s -o %t && %run %t

#include <assert.h>
#include <sanitizer/dfsan_interface.h>

typedef struct Pair {
  int i;
  char *ptr;
} Pair;

__attribute__((noinline))
Pair make_pair(int i, char *ptr) {
  Pair pair;
  pair.i = i;
  pair.ptr = ptr;
  return pair;
}

__attribute__((noinline))
Pair copy_pair1(const Pair *pair0) {
  Pair pair;
  pair.i = pair0->i;
  pair.ptr = pair0->ptr;
  return pair;
}

__attribute__((noinline))
Pair copy_pair2(const Pair pair0) {
  Pair pair;
  pair.i = pair0.i;
  pair.ptr = pair0.ptr;
  return pair;
}

int main(void) {
  int i = 1;
  char *ptr = NULL;
  dfsan_label i_label = dfsan_create_label("i", 0);
  dfsan_set_label(i_label, &i, sizeof(i));
  dfsan_label ptr_label = dfsan_create_label("ptr", 0);
  dfsan_set_label(ptr_label, &ptr, sizeof(ptr));

  Pair pair1 = make_pair(i, ptr);
  int i1 = pair1.i;
  char *ptr1 = pair1.ptr;

  dfsan_label i1_label = dfsan_read_label(&i1, sizeof(i1));
  dfsan_label ptr1_label = dfsan_read_label(&ptr1, sizeof(ptr1));
  assert(dfsan_has_label(i1_label, i_label));
  assert(dfsan_has_label(i1_label, ptr_label));
  assert(dfsan_has_label(ptr1_label, i_label));
  assert(dfsan_has_label(ptr1_label, ptr_label));

  Pair pair2 = copy_pair1(&pair1);
  int i2 = pair2.i;
  char *ptr2 = pair2.ptr;

  dfsan_label i2_label = dfsan_read_label(&i2, sizeof(i2));
  dfsan_label ptr2_label = dfsan_read_label(&ptr2, sizeof(ptr2));
  assert(dfsan_has_label(i2_label, i_label));
  assert(dfsan_has_label(i2_label, ptr_label));
  assert(dfsan_has_label(ptr2_label, i_label));
  assert(dfsan_has_label(ptr2_label, ptr_label));

  Pair pair3 = copy_pair2(pair1);
  int i3 = pair3.i;
  char *ptr3 = pair3.ptr;

  dfsan_label i3_label = dfsan_read_label(&i3, sizeof(i3));
  dfsan_label ptr3_label = dfsan_read_label(&ptr3, sizeof(ptr3));
  assert(dfsan_has_label(i3_label, i_label));
  assert(dfsan_has_label(i3_label, ptr_label));
  assert(dfsan_has_label(ptr3_label, i_label));
  assert(dfsan_has_label(ptr3_label, ptr_label));

  return 0;
}
