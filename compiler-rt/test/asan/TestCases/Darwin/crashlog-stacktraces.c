// RUN: %clang_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Since ASan is built with -fomit-frame-pointer, backtrace is not able to
// symbolicate the trace past ASan runtime on i386. (This is fixed in
// latest OS X.)
// REQUIRES: asan-64-bits

#include <execinfo.h>
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <stdlib.h>

void death_function() {
  fprintf(stderr, "DEATH CALLBACK\n");

  void* callstack[128];
  int i, frames = backtrace(callstack, 128);
  char** strs = backtrace_symbols(callstack, frames);
  for (i = 0; i < frames; ++i) {
    fprintf(stderr, "%s\n", strs[i]);
  }
  free(strs);

  fprintf(stderr, "END OF BACKTRACE\n");
}

int fault_function() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];  // BOOM
}

int main() {
  __sanitizer_set_death_callback(death_function);
  fault_function();
  return 0;
}

// CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
// CHECK: {{READ of size 1 at 0x.* thread T0}}
// CHECK: {{    #0 0x.* in fault_function}}

// CHECK: DEATH CALLBACK
// CHECK: death_function
// CHECK: fault_function
// CHECK: main
// CHECK: END OF BACKTRACE
