// Test that lsan reports a proper error when running under strace.
// REQUIRES: strace
// RUN: %clangxx_lsan %s -o %t
// RUN: not strace -o /dev/null %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

static volatile void *sink;

int main() {
  sink = malloc(42);
}
// CHECK: LeakSanitizer has encountered a fatal error
// CHECK: HINT: LeakSanitizer does not work under ptrace (strace, gdb, etc)
