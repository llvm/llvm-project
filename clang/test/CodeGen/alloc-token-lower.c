// Test optimization pipelines do not interfere with AllocToken lowering, and we
// pass on function attributes correctly.
//
// RUN: %clang_cc1     -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O2 -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

typedef __typeof(sizeof(int)) size_t;

void *malloc(size_t size);

// CHECK-LABEL: @test_malloc(
// CHECK: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 4, i64 0)
void *test_malloc() {
  return malloc(sizeof(int));
}

// CHECK-LABEL: @no_sanitize_malloc(
// CHECK: call{{.*}} ptr @malloc(i64 noundef 4)
void *no_sanitize_malloc(size_t size) __attribute__((no_sanitize("alloc-token"))) {
  return malloc(sizeof(int));
}
