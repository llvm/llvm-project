// Checks how we print the developer note "hwasan_dev_note_heap_rb_distance".
// RUN: %clang_hwasan %s -o %t
// RUN: not %run %t 10 2>&1 | FileCheck %s --check-prefix=D10
// RUN: not %run %t 42 2>&1 | FileCheck %s --check-prefix=D42

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>


void *p[100];

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  int distance = argc >= 2 ? atoi(argv[1]) : 1;
  for (int i = 0; i < 100; i++)
    p[i] = malloc(i);
  for (int i = 0; i < 100; i++)
    free(p[i]);

  *(int*)p[distance] = 0;
}

// D10: hwasan_dev_note_heap_rb_distance: 90 1023
// D42: hwasan_dev_note_heap_rb_distance: 58 1023
