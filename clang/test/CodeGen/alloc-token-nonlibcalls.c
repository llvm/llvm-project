// RUN: %clang_cc1    -fsanitize=alloc-token -fsanitize-alloc-token-extended -falloc-token-max=2147483647 -triple x86_64-linux-gnu -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O -fsanitize=alloc-token -fsanitize-alloc-token-extended -falloc-token-max=2147483647 -triple x86_64-linux-gnu -x c -emit-llvm %s -o - | FileCheck %s

typedef __typeof(sizeof(int)) size_t;
typedef size_t gfp_t;

void *custom_malloc(size_t size) __attribute__((malloc));
void *__kmalloc(size_t size, gfp_t flags) __attribute__((alloc_size(1)));

void *sink;

// CHECK-LABEL: @test_nonlibcall_alloc(
void test_nonlibcall_alloc() {
  // CHECK: call{{.*}} ptr @__alloc_token_custom_malloc(i64 noundef 4, i64 {{[1-9][0-9]*}})
  sink = custom_malloc(sizeof(int));
  // CHECK: call{{.*}} ptr @__alloc_token___kmalloc(i64 noundef 4, i64 noundef 0, i64 {{[1-9][0-9]*}})
  sink = __kmalloc(sizeof(int), 0);
}
