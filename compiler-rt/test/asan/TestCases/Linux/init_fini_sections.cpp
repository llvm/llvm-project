// RUN: %clangxx_asan %s -o %t && %run %t | FileCheck %s

#include <stdio.h>

int c = 0;

static void foo() {
  ++c;
}

static void fini() {
  printf("fini\n");
}

int main() {
  printf("c=%d\n", c);
  return 0;
}

__attribute__((section(".preinit_array")))
void (*call_foo)(void) = &foo;

__attribute__((section(".init_array")))
void (*call_foo_2)(void) = &foo;

__attribute__((section(".fini_array")))
void (*call_foo_3)(void) = &fini;

// CHECK: c=2
// CHECK: fini
