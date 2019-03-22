// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -O3 -disable-llvm-passes -o - %s | FileCheck %s

// Test that can call `__builtin_constant_p` with instances of different
// Objective-C classes.
// rdar://problem/47499250
@class Foo;
@class Bar;

extern void callee(void);

// CHECK-LABEL: define void @test(%0* %foo, %1* %bar)
void test(Foo *foo, Bar *bar) {
  // CHECK: [[ADDR_FOO:%.*]] = bitcast %0* %{{.*}} to i8*
  // CHECK-NEXT: call i1 @llvm.is.constant.p0i8(i8* [[ADDR_FOO]])
  // CHECK: [[ADDR_BAR:%.*]] = bitcast %1* %{{.*}} to i8*
  // CHECK-NEXT: call i1 @llvm.is.constant.p0i8(i8* [[ADDR_BAR]])
  if (__builtin_constant_p(foo) && __builtin_constant_p(bar))
    callee();
}

// Test other Objective-C types.
// CHECK-LABEL: define void @test_more(i8* %object, i8* %klass)
void test_more(id object, Class klass) {
  // CHECK: call i1 @llvm.is.constant.p0i8(i8* %{{.*}})
  // CHECK: call i1 @llvm.is.constant.p0i8(i8* %{{.*}})
  if (__builtin_constant_p(object) && __builtin_constant_p(klass))
    callee();
}
