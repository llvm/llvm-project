// RUN: %clang_cc1 %s -emit-llvm -o - -fms-extensions -triple i686-pc-win32 | FileCheck %s

// CHECK-LABEL: define dso_local void @test_alloca(
void capture(void *);
void test_alloca(int n) {
  capture(_alloca(n));
  // CHECK: %[[arg:.*]] = alloca i8, i32 %{{.*}}, align 16
  // CHECK: call void @capture(i8* %[[arg]])
}

// CHECK-LABEL: define dso_local void @test_alloca_with_align(
void test_alloca_with_align(int n) {
  capture(__builtin_alloca_with_align(n, 64));
  // CHECK: %[[arg:.*]] = alloca i8, i32 %{{.*}}, align 8
  // CHECK: call void @capture(i8* %[[arg]])
}
