// REQUIRES: binutils_lto

// RUN: %clang_pgogen=%t.profraw -fuse-ld=bfd -flto %s -o %t
// RUN: %run %t
// RUN: llvm-profdata merge %t.profraw -o %t.profdata
// RUN: llvm-profdata show %t.profdata | FileCheck %s

/// Test that we work around https://sourceware.org/bugzilla/show_bug.cgi?id=26262
/// (as of GNU ld 2.35) which happens when trying to generate IR profile with
/// BFD linker + LLVMgold.so

// CHECK: Instrumentation level: IR
int main() { return 0; }
