// RUN: %clangxx_asan -O %s -o %t
// RUN: not %run %t crash 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
// RUN: %env_asan_opts=poison_history_size=10000 not %run %t crash 2>&1 | FileCheck --check-prefix=CHECK-CRASH,POISON %s
// RUN: not %run %t bad-bounds 2>&1 | FileCheck --check-prefix=CHECK-BAD-BOUNDS %s
// RUN: not %run %t unaligned-bad-bounds 2>&1 | FileCheck --check-prefix=CHECK-UNALIGNED-BAD-BOUNDS %s --implicit-check-not="beg is not aligned by"
// RUN: not %run %t odd-alignment 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
// RUN: not %run %t odd-alignment-end 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
// RUN: %env_asan_opts=detect_container_overflow=0 %run %t crash
//
// RUN: not %run %t double-crash-beg 2>&1 | FileCheck --check-prefix=DOUBLE-CRASH-BEG %s
// RUN: not %run %t double-crash-end 2>&1 | FileCheck --check-prefix=DOUBLE-CRASH-END %s
// RUN: %env_asan_opts=poison_history_size=10000 not %run %t double-crash-beg 2>&1 | FileCheck --check-prefix=DOUBLE-CRASH-BEG,POISON %s
// RUN: %env_asan_opts=poison_history_size=10000 not %run %t double-crash-end 2>&1 | FileCheck --check-prefix=DOUBLE-CRASH-END,POISON %s
// RUN: not %run %t double-bad-bounds 2>&1 | FileCheck --check-prefix=DOUBLE-BAD-BOUNDS %s
// RUN: not %run %t double-unaligned-bad-bounds 2>&1 | FileCheck --check-prefix=DOUBLE-UNALIGNED-BAD-BOUNDS %s --implicit-check-not="beg is not aligned by"
// RUN: not %run %t double-odd-alignment 2>&1 | FileCheck --check-prefix=DOUBLE-CRASH-BEG %s
// RUN: not %run %t double-odd-alignment-end 2>&1 | FileCheck --check-prefix=DOUBLE-CRASH-END %s
// RUN: %env_asan_opts=detect_container_overflow=0 %run %t double-crash-beg
// RUN: %env_asan_opts=detect_container_overflow=0 %run %t double-crash-end
//
// Test crash due to __sanitizer_annotate_contiguous_container.

#include <assert.h>
#include <string.h>

#include <sanitizer/asan_interface.h>

static volatile int one = 1;

int TestCrash() {
  long t[100];
  t[60] = 0;
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 100,
                                            &t[0] + 50);
  // CHECK-CRASH: AddressSanitizer: container-overflow
  // CHECK-CRASH: if you don't care about these errors you may set ASAN_OPTIONS=detect_container_overflow=0
  return (int)t[60 * one]; // Touches the poisoned memory.
}

void BadBounds() {
  long t[100];
  // CHECK-BAD-BOUNDS: ERROR: AddressSanitizer: bad parameters to __sanitizer_annotate_contiguous_container
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 101,
                                            &t[0] + 50);
}

void UnalignedBadBounds() {
  char t[100];
  // CHECK-UNALIGNED-BAD-BOUNDS: ERROR: AddressSanitizer: bad parameters to __sanitizer_annotate_contiguous_container
  __sanitizer_annotate_contiguous_container(&t[1], &t[0] + 100, &t[0] + 101,
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

int DoubleEndedTestCrashBeg() {
  long t[100];
  t[15] = 0;
  __sanitizer_annotate_double_ended_contiguous_container(
      &t[0], &t[0] + 100, &t[0], &t[0] + 100, &t[0] + 25, &t[0] + 75);
  // DOUBLE-CRASH-BEG: AddressSanitizer: container-overflow
  // DOUBLE-CRASH-BEG: if you don't care about these errors you may set ASAN_OPTIONS=detect_container_overflow=0
  return (int)t[15 * one];
}

int DoubleEndedTestCrashEnd() {
  long t[100];
  t[85] = 0;
  __sanitizer_annotate_double_ended_contiguous_container(
      &t[0], &t[0] + 100, &t[0], &t[0] + 100, &t[0] + 25, &t[0] + 75);
  // DOUBLE-CRASH-END: AddressSanitizer: container-overflow
  // DOUBLE-CRASH-END: if you don't care about these errors you may set ASAN_OPTIONS=detect_container_overflow=0
  return (int)t[85 * one];
}

void DoubleEndedBadBounds() {
  long t[100];
  // DOUBLE-BAD-BOUNDS: ERROR: AddressSanitizer: bad parameters to __sanitizer_annotate_double_ended_contiguous_container
  __sanitizer_annotate_double_ended_contiguous_container(
      &t[0], &t[0] + 100, &t[0], &t[0] + 100, &t[0] + 75, &t[0] + 25);
}

void DoubleEndedUnalignedBadBounds() {
  char t[100];
  // DOUBLE-UNALIGNED-BAD-BOUNDS: ERROR: AddressSanitizer: bad parameters to __sanitizer_annotate_double_ended_contiguous_container
  __sanitizer_annotate_double_ended_contiguous_container(
      &t[1], &t[0] + 100, &t[0], &t[0] + 100, &t[0] + 25, &t[0] + 75);
}

int DoubleEndedOddAlignment() {
  int t[100];
  t[5] = 0;
  __sanitizer_annotate_double_ended_contiguous_container(
      &t[1], &t[0] + 100, &t[1], &t[0] + 100, &t[1] + 10, &t[1] + 60);
  // DOUBLE-CRASH-BEG: AddressSanitizer: container-overflow
  return (int)t[5 * one];
}

int DoubleEndedOddAlignmentEnd() {
  int t[100];
  t[95] = 0;
  __sanitizer_annotate_double_ended_contiguous_container(
      &t[0], &t[0] + 99, &t[0], &t[0] + 99, &t[0] + 10, &t[0] + 90);
  // DOUBLE-CRASH-END: AddressSanitizer: container-overflow
  return (int)t[95 * one];
}

// POISON: Memory was manually poisoned by thread T0:
// POISON: TestCrash

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "crash"))
    return TestCrash();
  else if (!strcmp(argv[1], "bad-bounds"))
    BadBounds();
  else if (!strcmp(argv[1], "unaligned-bad-bounds"))
    UnalignedBadBounds();
  else if (!strcmp(argv[1], "odd-alignment"))
    return OddAlignment();
  else if (!strcmp(argv[1], "odd-alignment-end"))
    return OddAlignmentEnd();
  else if (!strcmp(argv[1], "double-crash-beg"))
    return DoubleEndedTestCrashBeg();
  else if (!strcmp(argv[1], "double-crash-end"))
    return DoubleEndedTestCrashEnd();
  else if (!strcmp(argv[1], "double-bad-bounds"))
    DoubleEndedBadBounds();
  else if (!strcmp(argv[1], "double-unaligned-bad-bounds"))
    DoubleEndedUnalignedBadBounds();
  else if (!strcmp(argv[1], "double-odd-alignment"))
    return DoubleEndedOddAlignment();
  else if (!strcmp(argv[1], "double-odd-alignment-end"))
    return DoubleEndedOddAlignmentEnd();
  return 0;
}
