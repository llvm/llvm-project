// RUN: %clang_dfsan %s -mllvm -dfsan-combine-offset-labels-on-gep=false -o %t && %run %t
// RUN: %clang_dfsan %s -DPROP_OFFSET_LABELS -o %t && %run %t
//
// REQUIRES: x86_64-target-arch

// Tests that labels are propagated through GEP.

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

int main(void) {
  int i = 1;
  int *p = &i;
  int j = 2;
  // test that pointer arithmetic propagates labels in terms of the flag.
  dfsan_set_label(1, &i, sizeof(i));
  p += i;
#ifdef PROP_OFFSET_LABELS
  assert(dfsan_get_label(p) == 1);
#else
  assert(dfsan_get_label(p) == 0);
#endif
  // test that non-pointer operations always propagate labels.
  dfsan_set_label(2, &j, sizeof(j));
  j += i;
  assert(dfsan_get_label(j) == 3);
  return 0;
}
