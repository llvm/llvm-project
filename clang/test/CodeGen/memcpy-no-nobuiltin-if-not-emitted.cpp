// RUN: %clang_cc1 -triple i686-linux-gnu -std=c++11 -S -emit-llvm -o - %s | FileCheck %s
//
// Regression test for the issue reported at
// https://reviews.llvm.org/D78162#1986104

typedef unsigned long size_t;

extern "C" __inline__ __attribute__((__gnu_inline__)) void *memcpy(void *a, const void *b, unsigned c) {
  return __builtin_memcpy(a, b, c);
}
void *memcpy(void *, const void *, unsigned);

// CHECK-LABEL: define{{.*}} void @_Z1av
void a() { (void)memcpy; }

// CHECK-NOT: nobuiltin
