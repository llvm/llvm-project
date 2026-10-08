// RUN: %clang_cc1 -O -fsanitize=alloc-token -fsanitize-alloc-token-extended -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --implicit-check-not=__alloc_token
// RUN: %clang_cc1 -O -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --implicit-check-not=__alloc_token

typedef __typeof(sizeof(int)) size_t;

void *sink;

void *malloc(size_t size);
void *external_ptr_helper(void);
void external_void_helper(void);

__attribute__((malloc, always_inline)) inline void *wrapper(size_t size) {
  sink = external_ptr_helper();
  external_void_helper();
  return malloc(size);
}

// CHECK-LABEL: @test_inlined_wrapper(
// CHECK: call ptr @external_ptr_helper()
// CHECK: call void @external_void_helper()
// CHECK: call{{.*}} @__alloc_token_malloc(i64 noundef 4, i64 2689373973731826898){{.*}} !alloc_token [[META_INT:![0-9]+]]
void test_inlined_wrapper(void) {
  sink = wrapper(sizeof(int));
}

// CHECK: declare{{.*}} @__alloc_token_malloc(
// CHECK: [[META_INT]] = !{!"int", i1 false}
