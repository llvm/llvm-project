// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <sanitizer/msan_interface.h>

int main(void) {
  char *p = new char[16];
  __msan_dump_shadow(p, 5);
  delete[] p;
  const char *q = "abc";
  __msan_dump_shadow(q, 3);
  return 0;
}

// CHECK: ff ff ff ff ff
// CHECK: 00 00 00
