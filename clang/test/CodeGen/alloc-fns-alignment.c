// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s

// Note: this test originally asserted that malloc/calloc/realloc got alignment
// attributes on their return pointer. However, that was reverted in
// https://reviews.llvm.org/D118804 and it now asserts that they do _NOT_ get
// align attributes.

typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void *aligned_alloc(size_t, size_t);
void *memalign(size_t, size_t);

void *malloc_test(size_t n) {
  return malloc(n);
}

void *calloc_test(size_t n) {
  return calloc(1, n);
}

void *realloc_test(void *p, size_t n) {
  return realloc(p, n);
}

void *aligned_alloc_variable_test(size_t n, size_t a) {
  return aligned_alloc(a, n);
}

void *memalign_variable_test(size_t n, size_t a) {
  return memalign(a, n);
}

void *aligned_alloc_constant_test(size_t n) {
  return aligned_alloc(8, n);
}

void *aligned_alloc_large_constant_test(size_t n) {
  return aligned_alloc(4096, n);
}

void *memalign_large_constant_test(size_t n) {
  return memalign(4096, n);
}

// CHECK-LABEL: @malloc_test
// CHECK: call ptr @malloc

// CHECK: declare ptr @malloc

// CHECK-LABEL: @calloc_test
// CHECK: call ptr @calloc

// CHECK: declare ptr @calloc

// CHECK-LABEL: @realloc_test
// CHECK: call ptr @realloc

// CHECK: declare ptr @realloc

// CHECK-LABEL: @aligned_alloc_variable_test
// CHECK:      %[[ALLOCATED:.*]] = call ptr @aligned_alloc({{i32|i64}} noundef %[[ALIGN:.*]], {{i32|i64}} noundef %[[NBYTES:.*]])
// CHECK-NEXT: call void @llvm.assume(i1 true) [ "align"(ptr %[[ALLOCATED]], {{i32|i64}} %[[ALIGN]]) ]

// CHECK: declare ptr @aligned_alloc

// CHECK-LABEL: @memalign_variable_test
// CHECK:      %[[ALLOCATED:.*]] = call ptr @memalign({{i32|i64}} noundef %[[ALIGN:.*]], {{i32|i64}} noundef %[[NBYTES:.*]])
// CHECK-NEXT: call void @llvm.assume(i1 true) [ "align"(ptr %[[ALLOCATED]], {{i32|i64}} %[[ALIGN]]) ]

// CHECK-LABEL: @aligned_alloc_constant_test
// CHECK: call align 8 ptr @aligned_alloc

// CHECK-LABEL: @aligned_alloc_large_constant_test
// CHECK: call align 4096 ptr @aligned_alloc

// CHECK-LABEL: @memalign_large_constant_test
// CHECK: align 4096 ptr @memalign

