// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=use_registers=0:use_stacks=0 not %run %t 2>&1 | FileCheck %s

// Fixme: remove once test passes with hwasan
// UNSUPPORTED: hwasan

#include <stdio.h>
#include <stdlib.h>

#include "sanitizer/lsan_interface.h"

extern "C"
const char *__lsan_default_suppressions() {
  return "leak:*LSanTestLeakingFunc*";
}

void LSanTestLeakingFunc() {
  void *p = malloc(666);
  fprintf(stderr, "Test alloc: %p.\n", p);
}

int main() {
  LSanTestLeakingFunc();
  void *q = malloc(1337);
  fprintf(stderr, "Test alloc: %p.\n", q);
  return 0;
}
// CHECK: Suppressions used:
// CHECK: 1 666 *LSanTestLeakingFunc*
// CHECK: SUMMARY: {{(Leak|Address)}}Sanitizer: 1337 byte(s) leaked in 1 allocation(s)
