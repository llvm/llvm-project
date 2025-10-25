// RUN: %clang_cc1 -fsanitize=alloc-token -triple x86_64-linux-gnu -std=c++20 -fexceptions -fcxx-exceptions -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

#include "../Analysis/Inputs/system-header-simulator-cxx.h"
extern "C" {
void *aligned_alloc(size_t alignment, size_t size) __attribute__((malloc));
void *malloc(size_t size) __attribute__((malloc));
void *calloc(size_t num, size_t size) __attribute__((malloc));
void *realloc(void *ptr, size_t size) __attribute__((malloc));
void *reallocarray(void *ptr, size_t nmemb, size_t size) __attribute__((malloc));
void *memalign(size_t alignment, size_t size) __attribute__((malloc));
void *valloc(size_t size) __attribute__((malloc));
void *pvalloc(size_t size) __attribute__((malloc));
int posix_memalign(void **memptr, size_t alignment, size_t size);

struct __sized_ptr_t {
  void *p;
  size_t n;
};
enum class __hot_cold_t : uint8_t;
__sized_ptr_t __size_returning_new(size_t size);
__sized_ptr_t __size_returning_new_hot_cold(size_t, __hot_cold_t);
__sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t);
__sized_ptr_t __size_returning_new_aligned_hot_cold(size_t, std::align_val_t,  __hot_cold_t);
}

void *sink; // prevent optimizations from removing the calls

// CHECK-LABEL: define dso_local void @_Z16test_malloc_likev(
// CHECK: call noalias ptr @malloc(i64 noundef 4){{.*}} !alloc_token [[META_INT:![0-9]+]]
// CHECK: call noalias ptr @calloc(i64 noundef 3, i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @realloc(ptr noundef {{.*}}, i64 noundef 8){{.*}} !alloc_token [[META_LONG:![0-9]+]]
// CHECK: call noalias ptr @reallocarray(ptr noundef {{.*}}, i64 noundef 5, i64 noundef 8), !alloc_token [[META_LONG]]
// CHECK: call noalias align 128 ptr @aligned_alloc(i64 noundef 128, i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @memalign(i64 noundef 16, i64 noundef 4), !alloc_token [[META_INT]]
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

class ForwardDecl;

// CHECK-LABEL: define dso_local void @_Z21test_malloc_like_castv(
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_INT]]
// CHECK-NOT: call noalias ptr @malloc(i64 noundef 64){{.*}} !alloc_token [[META_INT]]
void test_malloc_like_cast() {
  sink = (int *)malloc(64);
  sink = reinterpret_cast<int *>(malloc(64));
  // Always fails to assign token ID for incomplete types.
  sink = reinterpret_cast<ForwardDecl *>(malloc(64));
}

// CHECK-LABEL: define dso_local void @_Z17test_operator_newv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4){{.*}} !alloc_token [[META_INT]]
void test_operator_new() {
  sink = __builtin_operator_new(sizeof(int));
  sink = ::operator new(sizeof(int));
}

// CHECK-LABEL: define dso_local void @_Z25test_operator_new_nothrowv(
// CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
// CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
void test_operator_new_nothrow() {
  sink = __builtin_operator_new(sizeof(int), std::nothrow);
  sink = ::operator new(sizeof(int), std::nothrow);
}

// CHECK-LABEL: define dso_local noundef ptr @_Z8test_newv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4){{.*}} !alloc_token [[META_INT]]
int *test_new() {
  return new int;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z14test_new_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 40){{.*}} !alloc_token [[META_INT]]
int *test_new_array() {
  return new int[10];
}

// CHECK-LABEL: define dso_local noundef ptr @_Z16test_new_nothrowv(
// CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
int *test_new_nothrow() {
  return new (std::nothrow) int;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z22test_new_array_nothrowv(
// CHECK: call noalias noundef ptr @_ZnamRKSt9nothrow_t(i64 noundef 40, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow){{.*}} !alloc_token [[META_INT]]
int *test_new_array_nothrow() {
  return new (std::nothrow) int[10];
}

// CHECK-LABEL: define dso_local void @_Z23test_size_returning_newv(
// CHECK: call { ptr, i64 } @__size_returning_new(i64 noundef 8)
// CHECK: call { ptr, i64 } @__size_returning_new_hot_cold(i64 noundef 8, i8 noundef zeroext 1)
// CHECK: call { ptr, i64 } @__size_returning_new_aligned(i64 noundef 8, i64 noundef 32)
// CHECK: call { ptr, i64 } @__size_returning_new_aligned_hot_cold(i64 noundef 8, i64 noundef 32, i8 noundef zeroext 1)
void test_size_returning_new() {
  // FIXME: Support __size_returning_new variants.
  sink = __size_returning_new(sizeof(long)).p;
  sink = __size_returning_new_hot_cold(sizeof(long), __hot_cold_t{1}).p;
  sink = __size_returning_new_aligned(sizeof(long), std::align_val_t{32}).p;
  sink = __size_returning_new_aligned_hot_cold(sizeof(long), std::align_val_t{32}, __hot_cold_t{1}).p;
}

class TestClass {
public:
  virtual void Foo();
  virtual ~TestClass();
  int data[16];
};

void may_throw();

// CHECK-LABEL: define dso_local noundef ptr @_Z27test_exception_handling_newv(
// CHECK: invoke noalias noundef nonnull ptr @_Znwm(i64 noundef 72)
// CHECK-NEXT: !alloc_token [[META_TESTCLASS:![0-9]+]]
TestClass *test_exception_handling_new() {
  try {
    TestClass *obj = new TestClass();
    may_throw();
    return obj;
  } catch (...) {
    return nullptr;
  }
}

// CHECK-LABEL: define dso_local noundef ptr @_Z14test_new_classv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 72){{.*}} !alloc_token [[META_TESTCLASS]]
TestClass *test_new_class() {
  TestClass *obj = new TestClass();
  obj->data[0] = 42;
  return obj;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z20test_new_class_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 728){{.*}} !alloc_token [[META_TESTCLASS]]
TestClass *test_new_class_array() {
  TestClass* arr = new TestClass[10];
  arr[0].data[0] = 123;
  return arr;
}

// CHECK: [[META_INT]] = !{!"int", i1 false}
// CHECK: [[META_LONG]] = !{!"long", i1 false}
// CHECK: [[META_TESTCLASS]] = !{!"TestClass", i1 true}
