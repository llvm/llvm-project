// RUN: %clangxx -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime

// XFAIL: ubsan
// FIXME: On Darwin, LSAn detects the leak, but does not invoke the death_callback.
// XFAIL: darwin && lsan

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
  // Trigger lsan report. Two attempts in case the address of the first
  // allocation remained on the stack.
  sink = new char[100];
  sink = new char[100];
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
