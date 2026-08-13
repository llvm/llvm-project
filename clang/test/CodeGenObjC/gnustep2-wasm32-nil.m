// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -emit-llvm -fobjc-runtime=gnustep-2.2 -o - %s | FileCheck %s


typedef struct {
  int x;
  int y;
  int z;
} S;

@interface Object
- (int)value;
- (S)s;
@end

// We expect generated nil-checks and zeroing of the result for WASM.

int sendToPossiblyNil(Object *object) {
  // CHECK-LABEL: define{{.*}} i32 @sendToPossiblyNil
  // CHECK: icmp eq ptr %{{.*}}, null
  // CHECK: br i1 %{{.*}}, label %[[CONTINUE:.*]], label %[[SEND:.*]]
  // CHECK: [[SEND]]:
  // CHECK: call ptr @objc_msg_lookup_sender
  // CHECK: br label %[[CONTINUE]]
  // CHECK: [[CONTINUE]]:
  // CHECK: phi i32 [ %{{.*}}, %[[SEND]] ], [ 0, %{{.*}} ]
  return [object value];
}

S sendStructToPossiblyNil(Object *object) {
  // CHECK-LABEL: define{{.*}} void @sendStructToPossiblyNil
  // CHECK: [[ISNIL:%.*]] = icmp eq ptr %{{.*}}, null
  // CHECK: br i1 [[ISNIL]], label %[[NIL_CLEANUP:.*]], label %[[STRUCT_SEND:.*]]
  // CHECK: [[STRUCT_SEND]]:
  // CHECK: call ptr @objc_msg_lookup_sender
  // CHECK: call void %{{.*}}(ptr{{.*}} sret(%struct.S){{.*}}
  // CHECK: br label %[[STRUCT_CONTINUE:.*]]
  // CHECK: [[NIL_CLEANUP]]:
  // CHECK-NEXT: call void @llvm.memset.p0.i32(ptr align 4 %agg.result, i8 0, i32 12, i1 false)
  // CHECK-NEXT: br label %[[STRUCT_CONTINUE]]
  // CHECK: [[STRUCT_CONTINUE]]:
  // CHECK: ret void
  return [object s];
}
