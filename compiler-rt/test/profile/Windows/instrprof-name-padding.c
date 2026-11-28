// RUN: %clang_profgen -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --counts --function=main %t.profraw | FileCheck %s

#pragma section(".lprfn$B", read)
__declspec(allocate(".lprfn$B")) const char Padding[3] = {0};

int main(void) { return 0; }

// CHECK: main:
// CHECK: Function count: 1
