// Check that PLT optimization works on X86-64 without -znow.

// RUN: %clang %cflags %s -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt --plt=all
// RUN: %t.bolt | FileCheck %s

// CHECK: Success

#include <stdio.h>

int main() {
  puts("Success");
  return 0;
}
