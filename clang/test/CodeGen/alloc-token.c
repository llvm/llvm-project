// RUN: %clang_cc1    -fsanitize=alloc-token -falloc-token-max=2147483647 -triple x86_64-linux-gnu -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O -fsanitize=alloc-token -falloc-token-max=2147483647 -triple x86_64-linux-gnu -x c -emit-llvm %s -o - | FileCheck %s

typedef __typeof(sizeof(int)) size_t;

void *aligned_alloc(size_t alignment, size_t size) __attribute__((malloc));
void *malloc(size_t size) __attribute__((malloc));
void *calloc(size_t num, size_t size) __attribute__((malloc));
void *realloc(void *ptr, size_t size) __attribute__((malloc));
void *reallocarray(void *ptr, size_t nmemb, size_t size) __attribute__((malloc));
void *memalign(size_t alignment, size_t size) __attribute__((malloc));
void *valloc(size_t size) __attribute__((malloc));
void *pvalloc(size_t size) __attribute__((malloc));
int posix_memalign(void **memptr, size_t alignment, size_t size);

void *sink;

// CHECK-LABEL: @test_malloc_like(
void test_malloc_like() {
  // CHECK: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 4, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token
  sink = malloc(sizeof(int));
  // CHECK: call{{.*}} ptr @__alloc_token_calloc(i64 noundef 3, i64 noundef 4, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token
  sink = calloc(3, sizeof(int));
  // CHECK: call{{.*}} ptr @__alloc_token_realloc(ptr noundef {{[^,]*}}, i64 noundef 8, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token
  sink = realloc(sink, sizeof(long));
  // CHECK: call{{.*}} ptr @__alloc_token_reallocarray(ptr noundef {{[^,]*}}, i64 noundef 5, i64 noundef 8, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token
  sink = reallocarray(sink, 5, sizeof(long));
  // CHECK: call{{.*}} align 128{{.*}} ptr @__alloc_token_aligned_alloc(i64 noundef 128, i64 noundef 4, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token
  sink = aligned_alloc(128, sizeof(int));
  // CHECK: call{{.*}} align 16{{.*}} ptr @__alloc_token_memalign(i64 noundef 16, i64 noundef 4, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token
  sink = memalign(16, sizeof(int));
  // CHECK: call{{.*}} ptr @__alloc_token_valloc(i64 noundef 4, i64 {{[1-9][0-9]*}}), !alloc_token
  sink = valloc(sizeof(int));
  // CHECK: call{{.*}} ptr @__alloc_token_pvalloc(i64 noundef 4, i64 {{[1-9][0-9]*}}), !alloc_token
  sink = pvalloc(sizeof(int));
  // FIXME: Should not be token ID 0!
  // CHECK: call{{.*}} i32 @__alloc_token_posix_memalign(ptr noundef {{[^,]*}}, i64 noundef 64, i64 noundef 4, i64 0)
  posix_memalign(&sink, 64, sizeof(int));
}

// CHECK-LABEL: @no_sanitize_malloc(
void *no_sanitize_malloc(size_t size) __attribute__((no_sanitize("alloc-token"))) {
  // CHECK: call{{.*}} ptr @malloc(
  return malloc(size);
}
