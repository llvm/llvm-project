// REQUIRES: darwin

// RUN: echo "static void dead_code(void) {}" > %t.dso.c
// RUN: %clang_profgen -fcoverage-mapping -O3 -dynamiclib -o %t.dso.dylib %t.dso.c
// RUN: %clang_profgen -fcoverage-mapping -O3 -o %t.exe %s %t.dso.dylib
// RUN: env LLVM_PROFILE_FILE="%c%t.profraw" %run %t.exe 2>&1 | count 0
// RUN: llvm-profdata show --counts --all-functions %t.profraw | FileCheck %s

// CHECK: Total functions: 1

int main() {}
