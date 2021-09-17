// RUN: %clang_dfsan %s -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -DFORCE_ZERO_LABELS -o %t && %run %t
// RUN: %clang_dfsan %s -o %t && %run %t
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

#include <assert.h>

int function_to_force_zero(int i, int* out) {
  *out = i;
  return i;
}

int main(void) {
  int i = 1;
  dfsan_label i_label = 2;
  dfsan_set_label(i_label, &i, sizeof(i));

  int out = 0;
  int ret = function_to_force_zero(i, &out);

#ifdef FORCE_ZERO_LABELS
  assert(dfsan_get_label(out) == 0);
  assert(dfsan_get_label(ret) == 0);
#else
  assert(dfsan_get_label(out) == i_label);
  assert(dfsan_get_label(ret) == i_label);
#endif

  return 0;
}
