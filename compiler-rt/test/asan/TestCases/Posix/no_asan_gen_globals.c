// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
// FIXME: http://llvm.org/bugs/show_bug.cgi?id=22682
// REQUIRES: asan-64-bits
// Make sure __asan_gen_* strings do not end up in the symbol table.

// RUN: %clang_asan %s -o %t.exe
// RUN: nm %t.exe | FileCheck %s

int x, y, z;
int main() { return 0; }
// CHECK-NOT: __asan_gen_
