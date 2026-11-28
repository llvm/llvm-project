// RUN: %clang_cc1 -triple arm64-apple-macosx -O0 -emit-llvm -fobjc-arc -fsanitize=thread -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -O1 -emit-llvm -fobjc-arc -fsanitize=thread -o - %s | FileCheck %s

extern id val;

// CHECK-LABEL: define{{.*}} void @test_store_strong(
// CHECK-NOT: call void @llvm.objc.storeStrong
// CHECK: call void @__tsan_write8(ptr [[DST:%.*]])
// CHECK-NEXT: store ptr {{.*}}, ptr [[DST]], align 8
// CHECK-NOT: call void @llvm.objc.storeStrong
// CHECK: ret void
void test_store_strong(__strong id *dst) {
  *dst = val;
}
