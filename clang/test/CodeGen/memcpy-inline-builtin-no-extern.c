// RUN: %clang_cc1 -triple x86_64 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// Verifies that clang-generated *.inline are flagged as internal.

typedef unsigned long size_t;

#define AVAILABLE_EXTERNALLY extern inline __attribute__((always_inline)) __attribute__((gnu_inline))
// Clang recognizes an inline builtin and renames it to memcmp.inline to prevent conflict with builtins.
AVAILABLE_EXTERNALLY int memcmp(const void *a, const void *b, size_t c) {
  return __builtin_memcmp(a, b, c);
}

// CHECK: internal{{.*}}memcmp.inline
int bar(const void *a, const void *b, size_t c) {
  return memcmp(a, b, c);
}

// Note that extern has been omitted here.
#define TRIPLE_INLINE inline __attribute__((always_inline)) __attribute__((gnu_inline))

// Clang recognizes an inline builtin and renames it to memcpy.inline to prevent conflict with builtins.
TRIPLE_INLINE void *memcpy(void *a, const void *b, size_t c) {
  return __builtin_memcpy(a, b, c);
}

// CHECK: internal{{.*}}memcpy.inline
void *foo(void *a, const void *b, size_t c) {
  return memcpy(a, b, c);
}
