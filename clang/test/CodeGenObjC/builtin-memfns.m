// RUN: %clang_cc1 -triple x86_64-apple-macosx10.8.0 -emit-llvm -o - %s | FileCheck %s

void *memcpy(void *restrict s1, const void *restrict s2, unsigned long n);

// PR13697
void test1(int *a, id b) {
  // CHECK: @test1
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 8, i1 false)
  memcpy(a, b, 8);
}
