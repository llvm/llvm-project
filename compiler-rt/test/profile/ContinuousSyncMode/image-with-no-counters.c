// REQUIRES: target={{.*(darwin|aix).*}}

// RUN: echo "static void dead_code(void) {}" > %t.dso.c
// RUN: %clang_profgen=%t.profraw -fprofile-continuous -fcoverage-mapping -O3 %shared_lib_flag -o %t.dso.dylib %t.dso.c
// RUN: %clang_profgen=%t.profraw -fprofile-continuous -fcoverage-mapping -O3 -o %t.exe %s %t.dso.dylib
// RUN: %run %t.exe 2>&1 | count 0
// RUN: llvm-profdata show --counts --all-functions %t.profraw | FileCheck %s

// CHECK: Total functions: 1

int main() {}
