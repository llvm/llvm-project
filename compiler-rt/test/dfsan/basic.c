// RUN: %clang_dfsan %s -o %t && %run %t
// RUN: %clang_dfsan -mllvm -dfsan-args-abi %s -o %t && %run %t
//
// REQUIRES: x86_64-target-arch

// Tests that labels are propagated through loads and stores.

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

int main(void) {
  int i = 1;
  dfsan_label i_label = dfsan_create_label("i", 0);
  dfsan_set_label(i_label, &i, sizeof(i));

  dfsan_label new_label = dfsan_get_label(i);
  assert(i_label == new_label);

  dfsan_label read_label = dfsan_read_label(&i, sizeof(i));
  assert(i_label == read_label);

  dfsan_label j_label = dfsan_create_label("j", 0);
  dfsan_add_label(j_label, &i, sizeof(i));

  read_label = dfsan_read_label(&i, sizeof(i));
  assert(dfsan_has_label(read_label, i_label));
  assert(dfsan_has_label(read_label, j_label));

  return 0;
}
