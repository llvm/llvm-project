// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

char *heap_ptr;

int main() {
  heap_ptr = (char *)malloc(10);
  fprintf(stderr, "heap_ptr: %p\n", heap_ptr);
  // CHECK: heap_ptr: 0x[[ADDR:[0-9a-f]+]]

  free(heap_ptr);
  free(heap_ptr);  // BOOM
  return 0;
}

void __asan_on_error() {
  int present = __asan_report_present();
  void *addr = __asan_get_report_address();
  const char *description = __asan_get_report_description();

  fprintf(stderr, "%s\n", (present == 1) ? "report present" : "");
  // CHECK: report present
  fprintf(stderr, "addr: %p\n", addr);
  // CHECK: addr: {{0x0*}}[[ADDR]]
  fprintf(stderr, "description: %s\n", description);
  // CHECK: description: double-free
}

// CHECK: AddressSanitizer: attempting double-free on {{0x0*}}[[ADDR]] in thread T0
