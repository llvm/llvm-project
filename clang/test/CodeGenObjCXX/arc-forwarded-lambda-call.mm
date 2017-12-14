// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -emit-llvm -disable-llvm-passes -O3 -fblocks -fobjc-arc -fobjc-runtime-has-weak -std=c++11 -o - %s | FileCheck %s

void test0(id x) {
  extern void test0_helper(id (^)(void));
  test0_helper([=]() { return x; });
  // CHECK-LABEL: define internal i8* @___Z5test0P11objc_object_block_invoke
  // CHECK: [[T0:%.*]] = call i8* @"_ZZ5test0P11objc_objectENK3$_0clEv"
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* [[T1]])
  // CHECK-NEXT: ret i8* [[T2]]
}

id test1_rv;

void test1() {
  extern void test1_helper(id (*)(void));
  test1_helper([](){ return test1_rv; });
  // CHECK-LABEL: define internal i8* @"_ZZ5test1vEN3$_18__invokeEv"
  // CHECK: [[T0:%.*]] = call i8* @"_ZZ5test1vENK3$_1clEv"
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* [[T1]])
  // CHECK-NEXT: ret i8* [[T2]]
}
