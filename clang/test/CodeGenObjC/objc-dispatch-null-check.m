// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-dispatch-method=non-legacy -fobjc-arc -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_STRONG:.*]] = type { ptr }

typedef struct {
  id x;
} Strong;

Strong getStrong(void);

@interface I0
- (void)passStrong:(Strong)a;
@end

// CHECK-LABEL: define{{.*}} void @test0(
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[CALL:.*]] = call ptr @getStrong()
// CHECK-NEXT: %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[AGG_TMP]], i32 0, i32 0
// CHECK-NEXT: store ptr %[[CALL]], ptr %[[COERCE_DIVE]], align 8

// CHECK: %[[MSGSEND_FN:.*]] = load ptr, ptr
// CHECK: %[[COERCE_DIVE1:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[AGG_TMP]], i32 0, i32 0
// CHECK: %[[V6:.*]] = load ptr, ptr %[[COERCE_DIVE1]], align 8
// CHECK: call void %[[MSGSEND_FN]]({{.*}}, ptr %[[V6]])
// CHECK: br

// CHECK: call void @__destructor_8_s0(ptr %[[AGG_TMP]])
// CHECK: br

void test0(I0 *a) {
  [a passStrong:getStrong()];
}
