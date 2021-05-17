// RUN: %clang_dfsan -DLIB -c %s -o %t.lib.o && \
// RUN: %clang_dfsan       -c %s -o %t.o && \
// RUN: %clang_dfsan %t.lib.o %t.o -o %t.bin && \
// RUN: %run %t.bin

// RUN: %clang_dfsan -mllvm -dfsan-args-abi -DLIB -c %s -o %t.lib.o && \
// RUN: %clang_dfsan -mllvm -dfsan-args-abi -c %s -o %t.o && \
// RUN: %clang_dfsan -mllvm -dfsan-args-abi %t.o %t.lib.o -o %t.bin && \
// RUN: %run %t.bin
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

#ifdef LIB
// Compiling this file with and without LIB defined allows this file to be
// built as two separate translation units.  This ensures that the code
// can not be optimized in a way that removes behavior we wish to test.  For
// example, computing a value should cause labels to be allocated only if
// the computation is actually done.  Putting the computation here prevents
// the compiler from optimizing away the computation (and labeling) that
// tests wish to verify.

int add_in_separate_translation_unit(int a, int b) {
  return a + b;
}

int multiply_in_separate_translation_unit(int a, int b) {
  return a * b;
}

#else

int add_in_separate_translation_unit(int i, int j);
int multiply_in_separate_translation_unit(int i, int j);

int main(void) {
  size_t label_count;

  // No labels allocated yet.
  label_count = dfsan_get_label_count();
  assert(0 == label_count);

  int i = 1;
  dfsan_label i_label = dfsan_create_label("i", 0);
  dfsan_set_label(i_label, &i, sizeof(i));

  // One label allocated for i.
  label_count = dfsan_get_label_count();
  assert(1u == label_count);

  int j = 2;
  dfsan_label j_label = dfsan_create_label("j", 0);
  dfsan_set_label(j_label, &j, sizeof(j));

  // Check that a new label was allocated for j.
  label_count = dfsan_get_label_count();
  assert(2u == label_count);

  // Create a value that combines i and j.
  int i_plus_j = add_in_separate_translation_unit(i, j);

  // Check that a label was created for the union of i and j.
  label_count = dfsan_get_label_count();
  assert(3u == label_count);

  // Combine i and j in a different way.  Check that the existing label is
  // reused, and a new label is not created.
  int j_times_i = multiply_in_separate_translation_unit(j, i);
  label_count = dfsan_get_label_count();
  assert(3u == label_count);
  assert(dfsan_get_label(i_plus_j) == dfsan_get_label(j_times_i));

  return 0;
}
#endif  // #ifdef LIB
