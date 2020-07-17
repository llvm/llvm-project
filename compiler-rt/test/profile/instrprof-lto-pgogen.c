// REQUIRES: lto
// XFAIL: msvc

// RUN: %clang_pgogen=%t.profraw -flto %s -o %t
// RUN: %run %t
// RUN: llvm-profdata merge %t.profraw -o %t.profdata
// RUN: llvm-profdata show %t.profdata | FileCheck %s

// Testing a bug that happens when trying to generate IR
// profile with BFD linker + LTO plugin

// CHECK: Instrumentation level: IR
int main() { return 0; }
