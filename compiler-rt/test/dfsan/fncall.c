// RUN: %clang_dfsan %s -o %t && %run %t
//
// Tests that labels are propagated through function calls.

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

int f(int x) {
  int j = 2;
  dfsan_label j_label = 2;
  dfsan_set_label(j_label, &j, sizeof(j));
  return x + j;
}

int main(void) {
  int i = 1;
  dfsan_label i_label = 4;
  dfsan_set_label(i_label, &i, sizeof(i));

  dfsan_label ij_label = dfsan_get_label(f(i));
  assert(dfsan_has_label(ij_label, i_label));

  /* Must be consistent with the one in f(). */
  dfsan_label j_label = 2;
  assert(dfsan_has_label(ij_label, 2));

  return 0;
}
