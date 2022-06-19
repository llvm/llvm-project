// RUN: %clang_cc1 -no-opaque-pointers -fno-builtin-aligned_alloc -emit-llvm < %s | FileCheck %s

typedef __SIZE_TYPE__ size_t;

void *aligned_alloc(size_t, size_t);

void *test(size_t alignment, size_t size) {
  // CHECK: call i8* @aligned_alloc{{.*}} #2
  return aligned_alloc(alignment, size);
}

// CHECK: attributes #2 = { nobuiltin "no-builtin-aligned_alloc" } 