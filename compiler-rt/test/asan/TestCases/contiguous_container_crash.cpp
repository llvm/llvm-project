// RUN: %clangxx_asan -O %s -o %t
// RUN: not %run %t crash 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
// RUN: not %run %t bad-bounds 2>&1 | FileCheck --check-prefix=CHECK-BAD-BOUNDS %s
// RUN: not %run %t odd-alignment 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
// RUN: not %run %t odd-alignment-end 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
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

int OddAlignment() {
  int t[100];
  t[60] = 0;
  __sanitizer_annotate_contiguous_container(&t[1], &t[0] + 100, &t[0] + 100,
                                            &t[1] + 50);
  return (int)t[60 * one]; // Touches the poisoned memory.
}

int OddAlignmentEnd() {
  int t[99];
  t[60] = 0;
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 98, &t[0] + 98,
                                            &t[0] + 50);
  return (int)t[60 * one]; // Touches the poisoned memory.
}

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "crash"))
    return TestCrash();
  else if (!strcmp(argv[1], "bad-bounds"))
    BadBounds();
  else if (!strcmp(argv[1], "odd-alignment"))
    return OddAlignment();
  else if (!strcmp(argv[1], "odd-alignment-end"))
    return OddAlignmentEnd();
  return 0;
}
