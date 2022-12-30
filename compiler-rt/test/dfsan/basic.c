// RUN: %clang_dfsan %s -o %t && %run %t

// Tests that labels are propagated through loads and stores.

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

int main(void) {
  int i = 1;
  dfsan_label i_label = 1;
  dfsan_set_label(i_label, &i, sizeof(i));

  dfsan_label new_label = dfsan_get_label(i);
  assert(i_label == new_label);

  dfsan_label read_label = dfsan_read_label(&i, sizeof(i));
  assert(i_label == read_label);

  dfsan_label j_label = 2;
  dfsan_add_label(j_label, &i, sizeof(i));

  read_label = dfsan_read_label(&i, sizeof(i));
  assert(dfsan_has_label(read_label, i_label));
  assert(dfsan_has_label(read_label, j_label));

  return 0;
}
