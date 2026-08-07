// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fobjc-runtime=gnustep-2.2 -fobjc-arc -fexceptions -fobjc-exceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm -o - %s | FileCheck --enable-var-scope %s
__attribute__((objc_root_class)) @interface Object @end
extern void mayThrowObjC();

int arcRethrow(Object *value) {
  @try {
    mayThrowObjC();
  } @catch (id caught) {
    @throw;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @arcRethrow
// CHECK: invoke void @mayThrowObjC()
// CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind to caller
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr @__objc_id_type_info]
// CHECK: br i1 %{{.*}}, label %[[CATCH:.*]], label %[[RETHROW:.*]]
// CHECK: [[RETHROW]]:
// CHECK-NEXT: call void @llvm.wasm.rethrow()
// CHECK-NEXT: unreachable
// CHECK: [[INVOKE_CONT]]:
// CHECK: br label %{{.*}}
// CHECK: [[CATCH]]:
// CHECK: invoke void @__cxa_rethrow(){{.*}}[ "funclet"(token [[CATCHPAD]]) ]
// CHECK-NEXT: to label %unreachable unwind label
