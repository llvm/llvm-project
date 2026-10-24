// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks -fobjc-runtime=ios-11.0 -emit-llvm -o - %s | FileCheck %s

typedef struct {
  __weak id values[2];
} WeakArray;

WeakArray getWeakArray(void);

// CHECK-LABEL: define void @copyWeakArray(
// CHECK: call ptr @llvm.objc.loadWeakRetained
// CHECK: call ptr @llvm.objc.storeWeak
// CHECK: call void @llvm.objc.release

void copyWeakArray(WeakArray *dst, WeakArray *src) {
  *dst = *src;
}

// CHECK-LABEL: define void @moveWeakArray(
// CHECK: call ptr @llvm.objc.loadWeakRetained
// CHECK: call ptr @llvm.objc.storeWeak
// CHECK: call void @llvm.objc.destroyWeak
// CHECK: call void @llvm.objc.release

void moveWeakArray(WeakArray *dst) {
  *dst = getWeakArray();
}
