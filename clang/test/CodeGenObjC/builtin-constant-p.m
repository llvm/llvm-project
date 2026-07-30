// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -O3 -disable-llvm-passes -o - %s | FileCheck %s

// Test that can call `__builtin_constant_p` with instances of different
// Objective-C classes.
@class Foo;
@class Bar;

extern void callee(void);

// CHECK-LABEL: define{{.*}} void @test(ptr noundef %foo, ptr noundef %bar)
void test(Foo *foo, Bar *bar) {
  // CHECK: call i1 @llvm.is.constant.p0(ptr %{{.*}})
  // CHECK: call i1 @llvm.is.constant.p0(ptr %{{.*}})
  if (__builtin_constant_p(foo) && __builtin_constant_p(bar))
    callee();
}

// Test other Objective-C types.
// CHECK-LABEL: define{{.*}} void @test_more(ptr noundef %object, ptr noundef %klass)
void test_more(id object, Class klass) {
  // CHECK: call i1 @llvm.is.constant.p0(ptr %{{.*}})
  // CHECK: call i1 @llvm.is.constant.p0(ptr %{{.*}})
  if (__builtin_constant_p(object) && __builtin_constant_p(klass))
    callee();
}
