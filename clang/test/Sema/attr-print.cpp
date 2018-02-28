// RUN: %clang_cc1 %s -ast-print | FileCheck %s

// CHECK: void *as2(int, int) __attribute__((alloc_size(1, 2)));
void *as2(int, int) __attribute__((alloc_size(1, 2)));
// CHECK: void *as1(void *, int) __attribute__((alloc_size(2)));
void *as1(void *, int) __attribute__((alloc_size(2)));
