// RUN: %clangxx_asan -O %s -o %t
// RUN: not %run %t crash 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
// RUN: not %run %t bad-bounds 2>&1 | FileCheck --check-prefix=CHECK-BAD-BOUNDS %s
// RUN: not %run %t bad-alignment 2>&1 | FileCheck --check-prefix=CHECK-BAD-ALIGNMENT %s
// RUN: %env_asan_opts=detect_container_overflow=0 %run %t crash
//
// Test crash due to __sanitizer_annotate_contiguous_container.

#include <assert.h>
#include <string.h>

extern "C" {
void __sanitizer_annotate_contiguous_container(const void *beg, const void *end,
                                               const void *old_mid,
                                               const void *new_mid);
}  // extern "C"

static volatile int one = 1;

int TestCrash() {
  long t[100];
  t[60] = 0;
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 100,
                                            &t[0] + 50);
// CHECK-CRASH: AddressSanitizer: container-overflow
// CHECK-CRASH: if you don't care about these errors you may set ASAN_OPTIONS=detect_container_overflow=0
  return (int)t[60 * one];  // Touches the poisoned memory.
}

void BadBounds() {
  long t[100];
// CHECK-BAD-BOUNDS: ERROR: AddressSanitizer: bad parameters to __sanitizer_annotate_contiguous_container
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 101,
                                            &t[0] + 50);
}

void BadAlignment() {
  int t[100];
// CHECK-BAD-ALIGNMENT: ERROR: AddressSanitizer: bad parameters to __sanitizer_annotate_contiguous_container
// CHECK-BAD-ALIGNMENT: ERROR: beg is not aligned by 8
  __sanitizer_annotate_contiguous_container(&t[1], &t[0] + 100, &t[1] + 10,
                                            &t[0] + 50);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "crash"))
    return TestCrash();
  else if (!strcmp(argv[1], "bad-bounds"))
    BadBounds();
  else if (!strcmp(argv[1], "bad-alignment"))
    BadAlignment();
}
