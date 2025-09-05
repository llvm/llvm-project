// RUN: %clang_cc1    -fsanitize=alloc-token -falloc-token-max=2147483647 -triple x86_64-linux-gnu -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O -fsanitize=alloc-token -falloc-token-max=2147483647 -triple x86_64-linux-gnu -x c -emit-llvm %s -o - | FileCheck %s

typedef __typeof(sizeof(int)) size_t;

void *aligned_alloc(size_t alignment, size_t size);
void *malloc(size_t size);
void *calloc(size_t num, size_t size);
void *realloc(void *ptr, size_t size);
void *reallocarray(void *ptr, size_t nmemb, size_t size);
void *memalign(size_t alignment, size_t size);
void *valloc(size_t size);
void *pvalloc(size_t size);
int posix_memalign(void **memptr, size_t alignment, size_t size);

void *sink;

// CHECK-LABEL: @test_malloc_like(
void test_malloc_like() {
  // FIXME: Should not be token ID 0! Currently fail to infer the type.
  // CHECK: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 4, i64 0)
  sink = malloc(sizeof(int));
  // CHECK: call{{.*}} ptr @__alloc_token_calloc(i64 noundef 3, i64 noundef 4, i64 0)
  sink = calloc(3, sizeof(int));
  // CHECK: call{{.*}} ptr @__alloc_token_realloc(ptr noundef {{[^,]*}}, i64 noundef 8, i64 0)
  sink = realloc(sink, sizeof(long));
  // CHECK: call{{.*}} ptr @__alloc_token_reallocarray(ptr noundef {{[^,]*}}, i64 noundef 5, i64 noundef 8, i64 0)
  sink = reallocarray(sink, 5, sizeof(long));
  // CHECK: call{{.*}} i32 @__alloc_token_posix_memalign(ptr noundef {{[^,]*}}, i64 noundef 64, i64 noundef 4, i64 0)
  posix_memalign(&sink, 64, sizeof(int));
  // CHECK: call align 128{{.*}} ptr @__alloc_token_aligned_alloc(i64 noundef 128, i64 noundef 1024, i64 0)
  sink = aligned_alloc(128, 1024);
  // CHECK: call align 16{{.*}} ptr @__alloc_token_memalign(i64 noundef 16, i64 noundef 256, i64 0)
  sink = memalign(16, 256);
  // CHECK: call{{.*}} ptr @__alloc_token_valloc(i64 noundef 4096, i64 0)
  sink = valloc(4096);
  // CHECK: call{{.*}} ptr @__alloc_token_pvalloc(i64 noundef 8192, i64 0)
  sink = pvalloc(8192);
}

// CHECK-LABEL: @no_sanitize_malloc(
void *no_sanitize_malloc(size_t size) __attribute__((no_sanitize("alloc-token"))) {
  // CHECK: call ptr @malloc(
  return malloc(size);
}
