// RUN: %clang_cc1 -std=gnu11 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s

// Test that counted_by on void* in GNU mode treats void as having size 1 (byte count)

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

struct with_counted_by_void {
  int count;
  void* buf __counted_by(count);
};

struct with_sized_by_void {
  int size;
  void* buf __sized_by(size);
};

struct with_counted_by_int {
  int count;
  int* buf __counted_by(count);
};

// CHECK-LABEL: define dso_local {{.*}}@test_counted_by_void(
// CHECK:         %[[COUNT:.*]] = load i32, ptr %s
// CHECK:         %[[NARROW:.*]] = tail call i32 @llvm.smax.i32(i32 %[[COUNT]], i32 0)
// CHECK:         %[[ZEXT:.*]] = zext nneg i32 %[[NARROW]] to i64
// CHECK:         ret i64 %[[ZEXT]]
//
// Verify: counted_by on void* returns the count directly (count * 1 byte)
long long test_counted_by_void(struct with_counted_by_void *s) {
  return __builtin_dynamic_object_size(s->buf, 0);
}

// CHECK-LABEL: define dso_local {{.*}}@test_sized_by_void(
// CHECK:         %[[SIZE:.*]] = load i32, ptr %s
// CHECK:         %[[NARROW:.*]] = tail call i32 @llvm.smax.i32(i32 %[[SIZE]], i32 0)
// CHECK:         %[[ZEXT:.*]] = zext nneg i32 %[[NARROW]] to i64
// CHECK:         ret i64 %[[ZEXT]]
//
// Verify: sized_by on void* returns the size directly
long long test_sized_by_void(struct with_sized_by_void *s) {
  return __builtin_dynamic_object_size(s->buf, 0);
}

// CHECK-LABEL: define dso_local {{.*}}@test_counted_by_int(
// CHECK:         %[[COUNT:.*]] = load i32, ptr %s
// CHECK:         %[[SEXT:.*]] = sext i32 %[[COUNT]] to i64
// CHECK:         %[[SIZE:.*]] = shl nsw i64 %[[SEXT]], 2
// CHECK:         ret i64
//
// Verify: counted_by on int* returns count * sizeof(int) = count * 4
long long test_counted_by_int(struct with_counted_by_int *s) {
  return __builtin_dynamic_object_size(s->buf, 0);
}

// CHECK-LABEL: define dso_local ptr @test_void_ptr_arithmetic(
// CHECK:         %[[BUF:.*]] = load ptr, ptr
// CHECK:         %[[EXT:.*]] = sext i32 %offset to i64
// CHECK:         %[[PTR:.*]] = getelementptr inbounds i8, ptr %[[BUF]], i64 %[[EXT]]
// CHECK:         ret ptr %[[PTR]]
//
// Verify: pointer arithmetic on void* uses i8 (byte offsets), not i32 or other sizes
void* test_void_ptr_arithmetic(struct with_counted_by_void *s, int offset) {
  return s->buf + offset;  // GNU extension: void* arithmetic
}
