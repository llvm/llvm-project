// RUN: %clangxx -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

volatile char *zero = 0;

void Death() {
  fprintf(stderr, "DEATH CALLBACK EXECUTED\n");
}
// CHECK: DEATH CALLBACK EXECUTED

char global;
volatile char *sink;

__attribute__((noinline))
void MaybeInit(int *uninitialized) {
  if (zero)
    *uninitialized = 1;
}

__attribute__((noinline))
void Leak() {
  sink = new char[100];  // trigger lsan report.
}

int main(int argc, char **argv) {
  int uninitialized;
  __sanitizer_set_death_callback(Death);
  MaybeInit(&uninitialized);
  if (uninitialized)  // trigger msan report.
    global = 77;
  sink = new char[100];
  delete[] sink;
  global = sink[0];  // use-after-free: trigger asan/tsan report.
  Leak();
  sink = 0;
}
