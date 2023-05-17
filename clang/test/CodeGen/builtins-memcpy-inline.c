// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_0(ptr noundef %dst, ptr noundef %src)
void test_memcpy_inline_0(void *dst, const void *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 0, i1 false)
  __builtin_memcpy_inline(dst, src, 0);
}

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_1(ptr noundef %dst, ptr noundef %src)
void test_memcpy_inline_1(void *dst, const void *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 1, i1 false)
  __builtin_memcpy_inline(dst, src, 1);
}

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_4(ptr noundef %dst, ptr noundef %src)
void test_memcpy_inline_4(void *dst, const void *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 4, i1 false)
  __builtin_memcpy_inline(dst, src, 4);
}

// CHECK-LABEL: define{{.*}} void @test_memcpy_inline_aligned_buffers(ptr noundef %dst, ptr noundef %src)
void test_memcpy_inline_aligned_buffers(unsigned long long *dst, const unsigned long long *src) {
  // CHECK:   call void @llvm.memcpy.inline.p0.p0.i64(ptr align 8 %0, ptr align 8 %1, i64 4, i1 false)
  __builtin_memcpy_inline(dst, src, 4);
}
