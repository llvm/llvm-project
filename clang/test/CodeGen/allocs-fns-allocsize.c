// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s

typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void *aligned_alloc(size_t, size_t);
void *memalign(size_t, size_t);

void *malloc_test(size_t n) {
  return malloc(n);
}

void *calloc_test(size_t e, size_t n) {
  return calloc(e, n);
}

void *realloc_test(void *p, size_t n) {
  return realloc(p, n);
}

void *aligned_alloc_test(size_t n, size_t a) {
  return aligned_alloc(a, n);
}

void *memalign_test(size_t n, size_t a) {
  return memalign(a, n);
}

// CHECK: @malloc(i64 noundef) #1
// CHECK: @calloc(i64 noundef, i64 noundef) #2
// CHECK: @realloc(i8* noundef, i64 noundef) #3
// CHECK: @aligned_alloc(i64 noundef, i64 noundef) #3
// CHECK: @memalign(i64 noundef, i64 noundef) #3

// CHECK: attributes #1 = { allocsize(0)
// CHECK: attributes #2 = { allocsize(0,1)
// CHECK: attributes #3 = { allocsize(1)
