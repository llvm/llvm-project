// Make sure ___asan_gen_* strings have the correct prefixes on Darwin
// ("L" in __TEXT,__cstring, "l" in __TEXT,__const

// RUN: %clang_asan %s -S -o %t.s
// RUN: cat %t.s | FileCheck %s || exit 1

// We test x86_64-specific peculiarities of ld on Darwin.
// REQUIRES: x86_64-target-arch

int x, y, z;
int main() { return 0; }
// CHECK: .section{{.*}}__TEXT,__const
// CHECK: l____asan_gen_
// CHECK: .section{{.*}}__TEXT,__cstring,cstring_literals
// CHECK: L____asan_gen_
// CHECK: L____asan_gen_
// CHECK: L____asan_gen_
