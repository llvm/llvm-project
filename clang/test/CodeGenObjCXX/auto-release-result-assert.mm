// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

// CHECK-LABEL: define{{.*}} ptr @_Z4foo1i(
// CHECK: %call1 = call noundef ptr @_Z4foo0i(i32 noundef %0) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK: call void (...) @llvm.objc.clang.arc.noop.use(ptr %call1) #3
// CHECK: %1 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call1) #3
// CHECK: ret ptr %1

// CHECK-LABEL: define{{.*}} ptr @_ZN2S22m1Ev(
// CHECK: %call2 = call noundef ptr @_Z4foo0i(i32 noundef 0) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK: call void (...) @llvm.objc.clang.arc.noop.use(ptr %call2) #3
// CHECK: %0 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call2) #3
// CHECK: ret ptr %0

// CHECK-LABEL: define internal noundef ptr @Block1_block_invoke(
// CHECK: %call1 = call noundef ptr @_Z4foo0i(i32 noundef 0) [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK: call void (...) @llvm.objc.clang.arc.noop.use(ptr %call1) #3
// CHECK: %0 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %call1) #3
// CHECK: ret ptr %0

struct S1;

typedef __attribute__((NSObject)) struct __attribute__((objc_bridge(id))) S1 * S1Ref;

S1Ref foo0(int);

struct S2 {
  S1Ref m1();
};

S1Ref foo1(int a) {
  return foo0(a);
}

S1Ref S2::m1() {
  return foo0(0);
}

S1Ref (^Block1)(void) = ^{
  return foo0(0);
};
