// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -emit-llvm -disable-llvm-passes -O3 -fblocks -fobjc-arc -fobjc-runtime-has-weak -std=c++11 -o - %s | FileCheck %s

void test0(id x) {
  extern void test0_helper(id (^)(void));
  test0_helper([=]() { return x; });
  // CHECK-LABEL: define internal i8* @___Z5test0P11objc_object_block_invoke
  // CHECK: [[T0:%.*]] = call i8* @"_ZZ5test0P11objc_objectENK3$_0clEv"{{.*}} [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* [[T0]])
  // CHECK-NEXT: ret i8* [[T2]]
}

// Check that the delegating block invoke function doesn't destruct the Weak
// object that is passed.

// CHECK-LABEL: define internal void @___Z8testWeakv_block_invoke(
// CHECK: call void @"_ZZ8testWeakvENK3$_2clE4Weak"(
// CHECK-NEXT: ret void

// CHECK-LABEL: define internal void @"_ZZ8testWeakvENK3$_2clE4Weak"(
// CHECK: call void @_ZN4WeakD1Ev(
// CHECK-NEXT: ret void

id test1_rv;

void test1() {
  extern void test1_helper(id (*)(void));
  test1_helper([](){ return test1_rv; });
  // CHECK-LABEL: define internal i8* @"_ZZ5test1vEN3$_18__invokeEv"
  // CHECK: [[T0:%.*]] = call i8* @"_ZZ5test1vENK3$_1clEv"{{.*}} [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* [[T0]])
  // CHECK-NEXT: ret i8* [[T2]]
}

struct Weak {
  __weak id x;
};

void testWeak() {
  extern void testWeak_helper(void (^)(Weak));
  testWeak_helper([](Weak){});
}
