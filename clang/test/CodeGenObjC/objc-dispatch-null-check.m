// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin9 -fobjc-dispatch-method=non-legacy -fobjc-arc -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_STRONG:.*]] = type { i8* }

typedef struct {
  id x;
} Strong;

Strong getStrong(void);

@interface I0
- (void)passStrong:(Strong)a;
@end

// CHECK-LABEL: define{{.*}} void @test0(
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[CALL:.*]] = call i8* @getStrong()
// CHECK-NEXT: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_STRONG]], %[[STRUCT_STRONG]]* %[[AGG_TMP]], i32 0, i32 0
// CHECK-NEXT: store i8* %[[CALL]], i8** %[[COERCE_DIVE]], align 8

// CHECK: %[[MSGSEND_FN:.*]] = load i8*, i8**
// CHECK: %[[V5:.*]] = bitcast i8* %[[MSGSEND_FN]] to void (i8*, i8*, i8*)*
// CHECK: %[[COERCE_DIVE1:.*]] = getelementptr inbounds %[[STRUCT_STRONG]], %[[STRUCT_STRONG]]* %[[AGG_TMP]], i32 0, i32 0
// CHECK: %[[V6:.*]] = load i8*, i8** %[[COERCE_DIVE1]], align 8
// CHECK: call void %[[V5]]({{.*}}, i8* %[[V6]])
// CHECK: br

// CHECK: %[[V7:.*]] = bitcast %[[STRUCT_STRONG]]* %[[AGG_TMP]] to i8**
// CHECK: call void @__destructor_8_s0(i8** %[[V7]])
// CHECK: br

void test0(I0 *a) {
  [a passStrong:getStrong()];
}
