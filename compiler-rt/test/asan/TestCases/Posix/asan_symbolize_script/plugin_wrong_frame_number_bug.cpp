// This test case checks for an old bug when using plug-ins that caused
// the stack numbering to be incorrect.
// UNSUPPORTED: android
// UNSUPPORTED: ios

// RUN: %clangxx_asan -O0 -g %s -o %t
// RUN: %env_asan_opts=symbolize=0 not %run %t DUMMY_ARG > %t.asan_report 2>&1
// RUN: %asan_symbolize --log-level debug --log-dest %t_debug_log_output.txt -l %t.asan_report --plugins %S/plugin_wrong_frame_number_bug.py > %t.asan_report_sym
// RUN: FileCheck --input-file=%t.asan_report_sym %s

#include <stdlib.h>

int* p;
extern "C" {

void bug() {
  free(p);
}

void foo(bool call_bug) {
  if (call_bug)
    bug();
}

// This indirection exists so that the call stack
// is reliably large enough.
void do_access_impl() {
  *p = 42;
}

void do_access() {
  do_access_impl();
}

int main(int argc, char** argv) {
  p = (int*) malloc(sizeof(p));
  foo(argc > 1);
  do_access();
  free(p);
  return 0;
}
}

// Check that the numbering of the stackframes is correct.

// CHECK: AddressSanitizer: heap-use-after-free
// CHECK-NEXT: WRITE of size
// CHECK-NEXT: #0 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: #1 0x{{[0-9a-fA-F]+}} in do_access
// CHECK-NEXT: #2 0x{{[0-9a-fA-F]+}} in main
