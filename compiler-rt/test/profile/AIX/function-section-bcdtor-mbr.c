// RUN: rm -f %t.profraw
// RUN: %clang_pgogen -ffunction-sections -Wl,-bcdtors:mbr %s -o %t.gen
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.gen
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s

int foo() { return 0; }
int main() { return foo();}

// CHECK: Total functions: 2
