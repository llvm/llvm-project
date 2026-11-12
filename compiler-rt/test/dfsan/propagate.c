// RUN: %clang_dfsan %s -o %t && %run %t

// Tests that labels are propagated through computation and that union labels
// are properly created.

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

int main(void) {
  assert(dfsan_union(0, 0) == 0);

  int i = 1;
  dfsan_label i_label = 1;
  dfsan_set_label(i_label, &i, sizeof(i));

  int j = 2;
  dfsan_label j_label = 2;
  dfsan_set_label(j_label, &j, sizeof(j));

  int k = 3;
  dfsan_label k_label = 4;
  dfsan_set_label(k_label, &k, sizeof(k));

  int k2 = 4;
  dfsan_set_label(k_label, &k2, sizeof(k2));

  dfsan_label ij_label = dfsan_get_label(i + j);
  assert(dfsan_has_label(ij_label, i_label));
  assert(dfsan_has_label(ij_label, j_label));
  assert(!dfsan_has_label(ij_label, k_label));
  // Test uniquing.
  assert(dfsan_union(i_label, j_label) == ij_label);
  assert(dfsan_union(j_label, i_label) == ij_label);

  dfsan_label ijk_label = dfsan_get_label(i + j + k);
  assert(dfsan_has_label(ijk_label, i_label));
  assert(dfsan_has_label(ijk_label, j_label));
  assert(dfsan_has_label(ijk_label, k_label));

  assert(dfsan_get_label(k + k2) == k_label);

  struct { int i, j; } s = { i, j };
  assert(dfsan_read_label(&s, sizeof(s)) == ij_label);

  return 0;
}
