// RUN: %clang_cc1 -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

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

// CHECK-LABEL: define dso_local void @test_malloc_like(
// CHECK: call ptr @malloc(i64 noundef 4)
// CHECK: call ptr @calloc(i64 noundef 3, i64 noundef 4)
// CHECK: call ptr @realloc(ptr noundef {{.*}}, i64 noundef 8)
// CHECK: call ptr @reallocarray(ptr noundef {{.*}}, i64 noundef 5, i64 noundef 8)
// CHECK: call align 128 ptr @aligned_alloc(i64 noundef 128, i64 noundef 1024)
// CHECK: call align 16 ptr @memalign(i64 noundef 16, i64 noundef 256)
// CHECK: call ptr @valloc(i64 noundef 4096)
// CHECK: call ptr @pvalloc(i64 noundef 8192)
// CHECK: call i32 @posix_memalign(ptr noundef @sink, i64 noundef 64, i64 noundef 4)
void test_malloc_like() {
  sink = malloc(sizeof(int));
  sink = calloc(3, sizeof(int));
  sink = realloc(sink, sizeof(long));
  sink = reallocarray(sink, 5, sizeof(long));
  sink = aligned_alloc(128, 1024);
  sink = memalign(16, 256);
  sink = valloc(4096);
  sink = pvalloc(8192);
  posix_memalign(&sink, 64, sizeof(int));
}
