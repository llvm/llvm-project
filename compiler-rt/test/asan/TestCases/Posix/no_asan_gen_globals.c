// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
// Bug 47607
// XFAIL: solaris
// Make sure ___asan_gen_* strings do not end up in the symbol table.

// RUN: %clang_asan %s -o %t.exe
// RUN: nm %t.exe | FileCheck %s

int x, y, z;
int main() { return 0; }
// CHECK-NOT: ___asan_gen_
