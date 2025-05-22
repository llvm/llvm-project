// REQUIRES: darwin || linux

// Test when LLVM_PROFILE_FILE is set incorrectly, it should fall backs to use default.profraw without runtime error.

// Create & cd into a temporary directory.
// RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir
// RUN: %clang -fprofile-instr-generate -fcoverage-mapping -mllvm -runtime-counter-relocation=true -o %t.exe %s
// RUN: env LLVM_PROFILE_FILE="incorrect-profile-name%m%c%c.profraw" %run %t.exe
// RUN: ls -l | FileCheck %s

// CHECK:     default.profraw
// CHECK-NOT: incorrect-profile-name.profraw

#include <stdio.h>
int f() { return 0; }

int main(int argc, char **argv) {
  FILE *File = fopen("default.profraw", "w");
  f();
  return 0;
}
