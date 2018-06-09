// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
// Make sure __asan_gen_* strings do not end up in the symbol table.

// RUN: %clang_asan %s -o %t.exe
// RUN: nm %t.exe | FileCheck %s

int x, y, z;
int main() { return 0; }
// CHECK-NOT: __asan_gen_
