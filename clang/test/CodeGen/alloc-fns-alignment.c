// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN16
// RUN: %clang_cc1 -triple x86_64-windows-msvc      -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN16
// RUN: %clang_cc1 -triple i386-apple-darwin        -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN16
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu   -emit-llvm < %s | FileCheck %s --check-prefix=ALIGN8
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-malloc  -emit-llvm < %s  | FileCheck %s --check-prefix=NOBUILTIN-MALLOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-calloc  -emit-llvm < %s  | FileCheck %s --check-prefix=NOBUILTIN-CALLOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-realloc -emit-llvm < %s  | FileCheck %s --check-prefix=NOBUILTIN-REALLOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-aligned_alloc -emit-llvm < %s  | FileCheck %s --check-prefix=NOBUILTIN-ALIGNED_ALLOC

typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void *aligned_alloc(size_t, size_t);

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

void *aligned_alloc_constant_test(size_t n) {
  return aligned_alloc(8, n);
}

void *aligned_alloc_large_constant_test(size_t n) {
  return aligned_alloc(4096, n);
}

// CHECK-LABEL: @malloc_test
// ALIGN16: align 16 i8* @malloc

// CHECK-LABEL: @calloc_test
// ALIGN16: align 16 i8* @calloc

// CHECK-LABEL: @realloc_test
// ALIGN16: align 16 i8* @realloc

// CHECK-LABEL: @aligned_alloc_variable_test
// ALIGN16:      %[[ALLOCATED:.*]] = call align 16 i8* @aligned_alloc({{i32|i64}} noundef %[[ALIGN:.*]], {{i32|i64}} noundef %[[NBYTES:.*]])
// ALIGN16-NEXT: call void @llvm.assume(i1 true) [ "align"(i8* %[[ALLOCATED]], {{i32|i64}} %[[ALIGN]]) ]

// CHECK-LABEL: @aligned_alloc_constant_test
// ALIGN16: align 16 i8* @aligned_alloc

// CHECK-LABEL: @aligned_alloc_large_constant_test
// ALIGN16: align 4096 i8* @aligned_alloc

// CHECK-LABEL: @malloc_test
// ALIGN8: align 8 i8* @malloc

// CHECK-LABEL: @calloc_test
// ALIGN8: align 8 i8* @calloc

// CHECK-LABEL: @realloc_test
// ALIGN8: align 8 i8* @realloc

// CHECK-LABEL: @aligned_alloc_variable_test
// ALIGN8: align 8 i8* @aligned_alloc

// CHECK-LABEL: @aligned_alloc_constant_test
// ALIGN8: align 8 i8* @aligned_alloc

// CHECK-LABEL: @aligned_alloc_large_constant_test
// ALIGN8: align 4096 i8* @aligned_alloc

// NOBUILTIN-MALLOC: declare i8* @malloc
// NOBUILTIN-CALLOC: declare i8* @calloc
// NOBUILTIN-REALLOC: declare i8* @realloc
// NOBUILTIN-ALIGNED_ALLOC: declare i8* @aligned_alloc
