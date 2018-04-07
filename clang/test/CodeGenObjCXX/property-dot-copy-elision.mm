// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -std=c++1z -fobjc-arc -o - %s | FileCheck %s

struct S0 {
  id f;
};

struct S1 {
  S1();
  S1(S0);
  id f;
};

@interface C
@property S1 f;
@end
@implementation C
@end

// CHECK-LABEL: define void @_Z5test0P1C(
// CHECK: %{{.*}} = alloca %
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_S1:.*]], align
// CHECK: %[[AGG_TMP_1:.*]] = alloca %[[STRUCT_S0:.*]], align
// CHECK: call void @_ZN2S0C1Ev(%[[STRUCT_S0]]* %[[AGG_TMP_1]])
// CHECK: call void @_ZN2S1C1E2S0(%[[STRUCT_S1]]* %[[AGG_TMP]], %[[STRUCT_S0]]* %[[AGG_TMP_1]])
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %[[STRUCT_S1]]*)*)(i8* %{{.*}}, i8* %{{.*}}, %[[STRUCT_S1]]* %[[AGG_TMP]])

void test0(C *c) {
  c.f = S0();
}

// CHECK: define void @_Z5test1P1C(
// CHECK: %{{.*}} = alloca %
// CHECK: %[[TEMP_LVALUE:.*]] = alloca %[[STRUCT_S1:.*]], align
// CHECK: call void @_ZN2S1C1Ev(%[[STRUCT_S1]]* %[[TEMP_LVALUE]])
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %[[STRUCT_S1]]*)*)(i8* %{{.*}}, i8* %{{.*}}, %[[STRUCT_S1]]* %[[TEMP_LVALUE]])

void test1(C *c) {
  c.f = S1();
}
