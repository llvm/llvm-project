// RUN: %clang_cc1    -fsanitize=alloc-token -falloc-token-max=2147483647 -triple x86_64-linux-gnu -std=c++20 -fexceptions -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O -fsanitize=alloc-token -falloc-token-max=2147483647 -triple x86_64-linux-gnu -std=c++20 -fexceptions -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s

#include "../Analysis/Inputs/system-header-simulator-cxx.h"
extern "C" {
void *aligned_alloc(size_t alignment, size_t size);
void *malloc(size_t size);
void *calloc(size_t num, size_t size);
void *realloc(void *ptr, size_t size);
void *reallocarray(void *ptr, size_t nmemb, size_t size);
void *memalign(size_t alignment, size_t size);
void *valloc(size_t size);
void *pvalloc(size_t size);
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

// CHECK-LABEL: @_Z16test_malloc_likev(
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
  // CHECK: call{{.*}} ptr @__alloc_token_memalign(i64 noundef 16, i64 noundef 256, i64 0)
  sink = memalign(16, 256);
  // CHECK: call{{.*}} ptr @__alloc_token_valloc(i64 noundef 4096, i64 0)
  sink = valloc(4096);
  // CHECK: call{{.*}} ptr @__alloc_token_pvalloc(i64 noundef 8192, i64 0)
  sink = pvalloc(8192);
}

// CHECK-LABEL: @_Z17test_operator_newv(
void test_operator_new() {
  // FIXME: This should not be token ID 0!
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 4, i64 0)
  sink = __builtin_operator_new(sizeof(int));
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 4, i64 0)
  sink = ::operator new(sizeof(int));
}

// CHECK-LABEL: @_Z25test_operator_new_nothrowv(
void test_operator_new_nothrow() {
  // FIXME: This should not be token ID 0!
  // CHECK: call {{.*}} ptr @__alloc_token_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr {{.*}} @_ZSt7nothrow, i64 0)
  sink = __builtin_operator_new(sizeof(int), std::nothrow);
  // CHECK: call {{.*}} ptr @__alloc_token_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr {{.*}} @_ZSt7nothrow, i64 0)
  sink = ::operator new(sizeof(int), std::nothrow);
}

// CHECK-LABEL: @_Z8test_newv(
int *test_new() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 4, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token_hint
  return new int;
}

// CHECK-LABEL: @_Z14test_new_arrayv(
int *test_new_array() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znam(i64 noundef 40, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token_hint
  return new int[10];
}

// CHECK-LABEL: @_Z16test_new_nothrowv(
int *test_new_nothrow() {
  // CHECK: call {{.*}} ptr @__alloc_token_ZnwmRKSt9nothrow_t(i64 noundef 4, ptr {{.*}} @_ZSt7nothrow, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token_hint
  return new (std::nothrow) int;
}

// CHECK-LABEL: @_Z22test_new_array_nothrowv(
int *test_new_array_nothrow() {
  // CHECK: call {{.*}} ptr @__alloc_token_ZnamRKSt9nothrow_t(i64 noundef 40, ptr {{.*}} @_ZSt7nothrow, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token_hint
  return new (std::nothrow) int[10];
}

// CHECK-LABEL: @_Z15no_sanitize_newv(
__attribute__((no_sanitize("alloc-token"))) int *no_sanitize_new() {
  // CHECK: call {{.*}} ptr @_Znwm(i64 noundef 4)
  return new int;
}

// CHECK-LABEL: @_Z23test_size_returning_newv(
void test_size_returning_new() {
  // FIXME: This should not be token ID 0!
  // CHECK: call { ptr, i64 } @__alloc_token_size_returning_new(i64 noundef 8, i64 0)
  // CHECK: call { ptr, i64 } @__alloc_token_size_returning_new_hot_cold(i64 noundef 8, i8 noundef zeroext 1, i64 0)
  // CHECK: call { ptr, i64 } @__alloc_token_size_returning_new_aligned(i64 noundef 8, i64 noundef 32, i64 0)
  // CHECK: call { ptr, i64 } @__alloc_token_size_returning_new_aligned_hot_cold(i64 noundef 8, i64 noundef 32, i8 noundef zeroext 1, i64 0)
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

// CHECK-LABEL: @_Z27test_exception_handling_newv(
TestClass *test_exception_handling_new() {
  try {
    // CHECK: invoke {{.*}} ptr @__alloc_token_Znwm(i64 noundef 72, i64 {{[1-9][0-9]*}})
    // CHECK-NEXT: !alloc_token_hint
    TestClass *obj = new TestClass();
    may_throw();
    return obj;
  } catch (...) {
    return nullptr;
  }
}

// CHECK-LABEL: @_Z14test_new_classv(
TestClass *test_new_class() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 72, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token_hint
  TestClass *obj = new TestClass();
  obj->data[0] = 42;
  return obj;
}

// CHECK-LABEL: @_Z20test_new_class_arrayv(
TestClass *test_new_class_array() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znam(i64 noundef 728, i64 {{[1-9][0-9]*}}){{.*}} !alloc_token_hint
  TestClass* arr = new TestClass[10];
  arr[0].data[0] = 123;
  return arr;
}

// CHECK-LABEL: @_Z21test_delete_unchangedPiS_(
void test_delete_unchanged(int *x, int *y) {
  // CHECK: call void @_ZdlPvm
  // CHECK: call void @_ZdaPv
  delete x;
  delete [] y;
}
