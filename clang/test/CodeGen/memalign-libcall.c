// RUN: %clang_cc1 -fno-builtin-memalign -emit-llvm < %s | FileCheck %s

typedef __SIZE_TYPE__ size_t;

void *memalign(size_t, size_t);

void *test(size_t alignment, size_t size) {
  // CHECK: call i8* @memalign{{.*}} #2
  return memalign(alignment, size);
}

// CHECK: attributes #2 = { nobuiltin "no-builtin-memalign" } 