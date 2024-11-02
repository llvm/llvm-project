// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_0(i8* noundef %dst, i8* noundef %src)
void test_memcpy_inline_0(void *dst, const void *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 0, i1 false)
  __builtin_memcpy_inline(dst, src, 0);
}

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_1(i8* noundef %dst, i8* noundef %src)
void test_memcpy_inline_1(void *dst, const void *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 1, i1 false)
  __builtin_memcpy_inline(dst, src, 1);
}

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_4(i8* noundef %dst, i8* noundef %src)
void test_memcpy_inline_4(void *dst, const void *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 4, i1 false)
  __builtin_memcpy_inline(dst, src, 4);
}

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_aligned_buffers(i64* noundef %dst, i64* noundef %src)
void test_memcpy_inline_aligned_buffers(unsigned long long *dst, const unsigned long long *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* align 8 %1, i8* align 8 %3, i64 4, i1 false) 
  __builtin_memcpy_inline(dst, src, 4);
}
