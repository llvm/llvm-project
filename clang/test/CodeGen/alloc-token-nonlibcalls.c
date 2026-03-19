// RUN: %clang_cc1    -fsanitize=alloc-token -fsanitize-alloc-token-extended -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes %s -o - | FileCheck --check-prefixes=CHECK,CHECK-CODEGEN %s
// RUN: %clang_cc1    -fsanitize=alloc-token -fsanitize-alloc-token-extended -triple x86_64-linux-gnu -emit-llvm                      %s -o - | FileCheck --check-prefixes=CHECK,CHECK-LOWER %s
// RUN: %clang_cc1 -O -fsanitize=alloc-token -fsanitize-alloc-token-extended -triple x86_64-linux-gnu -emit-llvm                      %s -o - | FileCheck --check-prefixes=CHECK,CHECK-LOWER %s

typedef __typeof(sizeof(int)) size_t;
typedef size_t gfp_t;

void *custom_malloc(size_t size) __attribute__((malloc));
void *__kmalloc(size_t size, gfp_t flags) __attribute__((alloc_size(1)));

void *sink;

// CHECK-LABEL: @test_nonlibcall_alloc(
// CHECK-CODEGEN: call noalias ptr @custom_malloc(i64 noundef 4){{.*}} !alloc_token [[META_INT:![0-9]+]]
// CHECK-CODEGEN: call ptr @__kmalloc(i64 noundef 4, i64 noundef 0){{.*}} !alloc_token [[META_INT]]
// CHECK-LOWER: call{{.*}} noalias ptr @__alloc_token_custom_malloc(i64 noundef 4, i64 2689373973731826898){{.*}} !alloc_token [[META_INT:![0-9]+]]
// CHECK-LOWER: call{{.*}} ptr @__alloc_token___kmalloc(i64 noundef 4, i64 noundef 0, i64 2689373973731826898){{.*}} !alloc_token [[META_INT]]
void test_nonlibcall_alloc() {
  sink = custom_malloc(sizeof(int));
  sink = __kmalloc(sizeof(int), 0);
}

// CHECK: [[META_INT]] = !{!"int", i1 false}
