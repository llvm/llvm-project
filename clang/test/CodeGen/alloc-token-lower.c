// Test optimization pipelines do not interfere with AllocToken lowering, and we
// pass on function attributes correctly.
//
// RUN: %clang_cc1     -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -O1 -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -O2 -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1     -fsanitize=alloc-token -fsanitize-alloc-token-fast-abi -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,FASTABI
// RUN: %clang_cc1 -O1 -fsanitize=alloc-token -fsanitize-alloc-token-fast-abi -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,FASTABI
// RUN: %clang_cc1 -O2 -fsanitize=alloc-token -fsanitize-alloc-token-fast-abi -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,FASTABI

typedef __typeof(sizeof(int)) size_t;

void *malloc(size_t size);

// CHECK-LABEL: @test_malloc(
// DEFAULT: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 4, i64 2689373973731826898){{.*}} !alloc_token [[META_INT:![0-9]+]]
// FASTABI: call{{.*}} ptr @__alloc_token_2689373973731826898_malloc(i64 noundef 4){{.*}} !alloc_token [[META_INT:![0-9]+]]
void *test_malloc() {
  return malloc(sizeof(int));
}

// CHECK-LABEL: @no_sanitize_malloc(
// CHECK: call{{.*}} ptr @malloc(i64 noundef 4)
void *no_sanitize_malloc(size_t size) __attribute__((no_sanitize("alloc-token"))) {
  return malloc(sizeof(int));
}

// By default, we should not be touching malloc-attributed non-libcall
// functions: there might be an arbitrary number of these, and a compatible
// allocator will only implement standard allocation functions.
void *nonstandard_malloc(size_t size) __attribute__((malloc));
// CHECK-LABEL: @test_nonlibcall_malloc(
// CHECK-NOT: __alloc_token_
// CHECK: call{{.*}} ptr @nonstandard_malloc(i64 noundef 4){{.*}} !alloc_token [[META_INT]]
void *test_nonlibcall_malloc() {
  return nonstandard_malloc(sizeof(int));
}

// CHECK: [[META_INT]] = !{!"int", i1 false}
