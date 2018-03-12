// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks -fobjc-runtime=ios-11.0 -fobjc-exceptions -fexceptions -fobjc-arc-exceptions -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_STRONG:.*]] = type { i32, i8* }

typedef struct {
  int i;
  id f1;
} Strong;

// CHECK: define void @testStrongException()
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[AGG_TMP1:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[CALL:.*]] = call [2 x i64] @genStrong()
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONG]]* %[[AGG_TMP]] to [2 x i64]*
// CHECK: store [2 x i64] %[[CALL]], [2 x i64]* %[[V0]], align 8
// CHECK: invoke [2 x i64] @genStrong()

// CHECK: call void @calleeStrong([2 x i64] %{{.*}}, [2 x i64] %{{.*}})
// CHECK-NEXT: ret void

// CHECK: landingpad { i8*, i32 }
// CHECK: %[[V9:.*]] = bitcast %[[STRUCT_STRONG]]* %[[AGG_TMP]] to i8**
// CHECK: call void @__destructor_8_s8(i8** %[[V9]])
// CHECK: br label

// CHECK: resume

Strong genStrong(void);
void calleeStrong(Strong, Strong);

void testStrongException(void) {
  calleeStrong(genStrong(), genStrong());
}
