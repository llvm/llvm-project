// RUN: %clang_dfsan %s -mllvm -dfsan-combine-offset-labels-on-gep=false -mllvm -dfsan-combine-pointer-labels-on-load=false -mllvm -dfsan-combine-taint-lookup-table=remap_to_upper -DLOOKUP_TABLE -o %t && %run %t
// RUN: %clang_dfsan %s -mllvm -dfsan-combine-offset-labels-on-gep=false -mllvm -dfsan-combine-pointer-labels-on-load=false -mllvm -dfsan-combine-taint-lookup-table=no_match -o %t && %run %t
// RUN: %clang_dfsan %s -mllvm -dfsan-combine-offset-labels-on-gep=false -mllvm -dfsan-combine-pointer-labels-on-load=false -o %t && %run %t
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

const char remap_to_upper[256] = {
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.',
};

char character_mapping(unsigned char c) {
  return remap_to_upper[c];
}

int main(void) {
  char a = 'b';
  dfsan_label i_label = 1;
  dfsan_set_label(i_label, &a, sizeof(a));
  assert(dfsan_read_label(&a, sizeof(a)) == i_label);

  char b = character_mapping(a);
  assert(b == 'B');

#ifdef LOOKUP_TABLE
  assert(dfsan_read_label(&b, sizeof(b)) == i_label);
#else
  assert(dfsan_read_label(&b, sizeof(b)) == 0);
#endif
  return 0;
}
