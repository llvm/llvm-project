// Check that when the program closed its std(in|out|err), running the external
// symbolizer still works.

// RUN: rm -f %t.log.*
// RUN: %clangxx_asan -O0 %s -o %t 2>&1 && %env_asan_opts=log_path='"%t.log"':verbosity=2 not %run %t 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-FILE < %t.log.*

// FIXME: copy %t.log back from the device and re-enable on Android.
// UNSUPPORTED: android

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  int result = fprintf(stderr, "Closing streams.\n");
  assert(result > 0);
  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);
  result = fprintf(stderr, "Can you hear me now?\n");
  assert(result < 0);
  char *x = (char *)malloc(10 * sizeof(char));
  free(x);
  x[argc] = 'X';  // BOOM
  // CHECK-FILE: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK-FILE:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
  // CHECK-FILE: {{WRITE of size 1 at 0x.* thread T0}}
  // CHECK-FILE: {{    #0 0x.* in main .*closed-fds.cc:}}[[@LINE-4]]
  return 0;
}
