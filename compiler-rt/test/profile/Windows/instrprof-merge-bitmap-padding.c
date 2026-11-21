// RUN: %clang_pgogen -O2 -o %t %s
// RUN: rm -f %t-*.profraw
// RUN: env LLVM_PROFILE_FILE=%t-%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t-%m.profraw %run %t
// RUN: llvm-profdata show --counts --function=foo %t-*.profraw | FileCheck %s

#pragma section(".lprfb$M", read, write)
__declspec(allocate(".lprfb$M")) char Bitmap[3] = {0};

int foo(void) { return 0; }
int (*volatile Target)(void) = foo;

int main(void) { return Target(); }

// CHECK: foo:
// CHECK: Block counts: [2]
