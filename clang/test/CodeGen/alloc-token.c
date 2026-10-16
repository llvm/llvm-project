// RUN: %clang_cc1 -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

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

// CHECK-LABEL: define dso_local void @test_malloc_like(
// CHECK: call noalias ptr @malloc(i64 noundef 4){{.*}} !alloc_token [[META_INT:![0-9]+]]
// CHECK: call noalias ptr @calloc(i64 noundef 3, i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @realloc(ptr noundef {{.*}}, i64 noundef 8){{.*}} !alloc_token [[META_LONG:![0-9]+]]
// CHECK: call noalias ptr @reallocarray(ptr noundef {{.*}}, i64 noundef 5, i64 noundef 8), !alloc_token [[META_LONG]]
// CHECK: call noalias align 128 ptr @aligned_alloc(i64 noundef 128, i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias align 16 ptr @memalign(i64 noundef 16, i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @valloc(i64 noundef 4), !alloc_token [[META_INT]]
// CHECK: call noalias ptr @pvalloc(i64 noundef 4), !alloc_token [[META_INT]]
// CHECK: call i32 @posix_memalign(ptr noundef @sink, i64 noundef 64, i64 noundef 4)
void test_malloc_like() {
  sink = malloc(sizeof(int));
  sink = calloc(3, sizeof(int));
  sink = realloc(sink, sizeof(long));
  sink = reallocarray(sink, 5, sizeof(long));
  sink = aligned_alloc(128, sizeof(int));
  sink = memalign(16, sizeof(int));
  sink = valloc(sizeof(int));
  sink = pvalloc(sizeof(int));
  posix_memalign(&sink, 64, sizeof(int)); // FIXME: support posix_memalign
}

// CHECK: [[META_INT]] = !{!"int", i1 false}
// CHECK: [[META_LONG]] = !{!"long", i1 false}
