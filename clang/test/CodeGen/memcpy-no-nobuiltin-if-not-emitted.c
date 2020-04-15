// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s
//
// Verifies that clang doesn't mark an inline builtin definition as `nobuiltin`
// if the builtin isn't emittable.

typedef unsigned long size_t;

// always_inline is used so clang will emit this body. Otherwise, we need >=
// -O1.
#define AVAILABLE_EXTERNALLY extern inline __attribute__((always_inline)) \
    __attribute__((gnu_inline))

AVAILABLE_EXTERNALLY void *memcpy(void *a, const void *b, size_t c) {
  return __builtin_memcpy(a, b, c);
}

// CHECK-LABEL: define void @foo
void foo(void *a, const void *b, size_t c) {
  // Clang will always _emit_ this as memcpy. LLVM turns it into @llvm.memcpy
  // later on if optimizations are enabled.
  // CHECK: call i8* @memcpy
  memcpy(a, b, c);
}

// CHECK-NOT: nobuiltin
