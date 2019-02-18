// RUN: %clang_cc1 %s -emit-llvm -o - -fobjc-gc -fblocks -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 | FileCheck %s --check-prefix=CHECK --check-prefix=OBJC
// RUN: %clang_cc1 -x objective-c++ %s -emit-llvm -o - -fobjc-gc -fblocks -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 | FileCheck %s --check-prefix=CHECK --check-prefix=OBJCXX

// OBJC-LABEL: define void @test1(
// OBJCXX-LABEL: define void @_Z5test1P12NSDictionary(

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_
// CHECK: call void @_Block_object_assign(

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_
// CHECK: call void @_Block_object_dispose(

// OBJC-LABEL: define void @foo(
// OBJCXX-LABEL: define void @_Z3foov(
// CHECK: call i8* @objc_read_weak(
// CHECK: call i8* @objc_assign_weak(
// CHECK: call void @_Block_object_dispose(

// OBJC-LABEL: define void @test2(
// OBJCXX-LABEL: define void @_Z5test2v(
// CHECK: call i8* @objc_assign_weak(
// CHECK: call void @_Block_object_dispose(

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_
// CHECK: call void @_Block_object_assign(

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_
// CHECK: call void @_Block_object_dispose(

@interface NSDictionary @end

void test1(NSDictionary * dict) {
  ^{ (void)dict; }();
}

@interface D
@end

void foo() {
  __block __weak D *weakSelf;
  ^{ (void)weakSelf; };
  D *l;
  l = weakSelf;
  weakSelf = l;
}

void (^__weak b)(void);

void test2() {
  __block int i = 0;
  b = ^ {  ++i; };
}
