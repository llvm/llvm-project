// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=array-bounds -emit-llvm -o - %s | FileCheck %s

// Test that counted_by on a function parameter is consumed entirely by Sema:
// unlike a field's count, it bounds neither -fsanitize=array-bounds nor
// __builtin_dynamic_object_size.

#define __counted_by(f)  __attribute__((counted_by(f)))

// CHECK-LABEL: define dso_local i32 @subscript(
// CHECK-NOT:     __ubsan_handle_out_of_bounds
// CHECK:         ret i32
//
// Verify: indexing a counted parameter is not bounds checked
int subscript(int count, int *__counted_by(count) buf, int i) {
  return buf[i];
}

// CHECK-LABEL: define dso_local i64 @bdos(
// CHECK-NOT:     counted_by.load
// CHECK:         call i64 @llvm.objectsize.i64.p0(
//
// Verify: the object size is not computed from the count
unsigned long bdos(int count, int *__counted_by(count) buf) {
  return __builtin_dynamic_object_size(buf, 0);
}
