// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fobjc-exceptions -fexceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm -fobjc-runtime=gnustep-2.2 -o - %s | FileCheck --enable-var-scope %s

__attribute__((objc_root_class)) @interface Object
@end

@interface ExceptionA : Object
@end

@interface ExceptionB : Object
@end

void mayThrow(void) {
  @throw (id)1;
}

int basicCatchAll(void) {
  @try {
    mayThrow();
  } @catch (...) {
    return 1;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @basicCatchAll
// CHECK: invoke void @mayThrow()
// CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind to caller
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr null]
// CHECK: br label %[[CATCH_ALL:.*]]
// CHECK: [[INVOKE_CONT]]:
// CHECK: br label %[[EH_CONT:.*]]
// CHECK: [[CATCH_ALL]]:
// CHECK: call ptr @__cxa_begin_catch
// CHECK: call void @__cxa_end_catch()
// CHECK: catchret from [[CATCHPAD]] to label %[[CATCHRET_DEST:.*]]
// CHECK: [[CATCHRET_DEST]]:
// CHECK-NEXT: br label %return

int twoTypedHandlers(void) {
  @try {
    mayThrow();
  } @catch (ExceptionA *exception) {
    return 1;
  } @catch (ExceptionB *exception) {
    return 2;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @twoTypedHandlers
// CHECK: invoke void @mayThrow()
// CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind to caller
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr @__objc_eh_typeinfo_ExceptionA, ptr @__objc_eh_typeinfo_ExceptionB]
// CHECK: br i1 %{{.*}}, label %[[CATCH:.*]], label %[[CATCH_FALLTHROUGH:.*]]
// CHECK: [[CATCH_FALLTHROUGH]]:
// CHECK: br i1 %{{.*}}, label %[[CATCH2:.*]], label %[[RETHROW:.*]]
// CHECK: [[RETHROW]]:
// CHECK-NEXT: call void @llvm.wasm.rethrow()
// CHECK-NEXT: unreachable
// CHECK: [[INVOKE_CONT]]:
// CHECK: br label %[[EH_CONT:.*]]
// CHECK: [[EH_CONT]]:
// CHECK: br label %return
// CHECK: [[CATCH]]:
// CHECK: catchret from [[CATCHPAD]] to label %[[CATCHRET_DEST:.*]]
// CHECK: [[CATCHRET_DEST]]:
// CHECK-NEXT: br label %return
// CHECK: [[CATCH2]]:
// CHECK: catchret from [[CATCHPAD]] to label %[[CATCHRET_DEST2:.*]]
// CHECK: [[CATCHRET_DEST2]]:
// CHECK-NEXT: br label %return

int typedHandlerAndCatchAll(void) {
  @try {
    mayThrow();
  } @catch (ExceptionA *exception) {
    return 1;
  } @catch (...) {
    return 2;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @typedHandlerAndCatchAll
// CHECK: invoke void @mayThrow()
// CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind to caller
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr @__objc_eh_typeinfo_ExceptionA, ptr null]
// CHECK: br i1 %{{.*}}, label %[[CATCH:.*]], label %[[CATCH_ALL:.*]]
// CHECK: [[INVOKE_CONT]]:
// CHECK: br label %[[EH_CONT:.*]]
// CHECK: [[EH_CONT]]:
// CHECK: br label %return
// CHECK: [[CATCH]]:
// CHECK: catchret from [[CATCHPAD]] to label %[[CATCHRET_DEST:.*]]
// CHECK: [[CATCHRET_DEST]]:
// CHECK-NEXT: br label %return
// CHECK: [[CATCH_ALL]]:
// CHECK: catchret from [[CATCHPAD]] to label %[[CATCHRET_DEST_ALL:.*]]
// CHECK: [[CATCHRET_DEST_ALL]]:
// CHECK-NEXT: br label %return

int nestedTryCatch(void) {
  @try {
    @try {
      mayThrow();
    } @catch (ExceptionA *exception) {
      return 1;
    }
  } @catch (...) {
    return 2;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @nestedTryCatch
// CHECK: invoke void @mayThrow()
// CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind label %[[CATCH_DISPATCH1:.*]]
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr @__objc_eh_typeinfo_ExceptionA]
// CHECK: br i1 %{{.*}}, label %[[CATCH:.*]], label %[[RETHROW:.*]]
// CHECK: [[RETHROW]]:
// CHECK: invoke void @llvm.wasm.rethrow(){{.*}}[ "funclet"(token [[CATCHPAD]]) ]
// CHECK-NEXT: to label %[[UNREACHABLE:.*]] unwind label %[[CATCH_DISPATCH1]]
// CHECK: [[CATCH_DISPATCH1]]:
// CHECK-NEXT: [[CATCHSWITCH1:%.*]] = catchswitch within none [label %[[CATCH_START2:.*]]] unwind to caller
// CHECK: [[CATCH_START2]]:
// CHECK-NEXT: [[CATCHPAD1:%.*]] = catchpad within [[CATCHSWITCH1]] [ptr null]
// CHECK: [[INVOKE_CONT]]:
// CHECK: br label %[[EH_CONT:.*]]
// CHECK: [[EH_CONT]]:
// CHECK: br label %[[EH_CONT2:.*]]
// CHECK: [[EH_CONT2]]:
// CHECK: br label %return
// CHECK: [[CATCH]]:
// CHECK: catchret from [[CATCHPAD]] to label %[[CATCHRET_DEST:.*]]
// CHECK: [[CATCHRET_DEST]]:
// CHECK-NEXT: br label %return
// CHECK: [[CATCH_ALL:.*]]:
// CHECK: catchret from [[CATCHPAD1]] to label %[[CATCHRET_DEST_ALL:.*]]
// CHECK: [[CATCHRET_DEST_ALL]]:
// CHECK-NEXT: br label %return

int emptyCatch(void) {
  @try {
    mayThrow();
  } @catch (ExceptionA *exception) {
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @emptyCatch
// CHECK: invoke void @mayThrow()
// CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind to caller
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr @__objc_eh_typeinfo_ExceptionA]
// CHECK: br i1 %{{.*}}, label %[[CATCH:.*]], label %[[RETHROW:.*]]
// CHECK: [[RETHROW]]:
// CHECK-NEXT: call void @llvm.wasm.rethrow()
// CHECK-NEXT: unreachable
// CHECK: [[INVOKE_CONT]]:
// CHECK: br label %[[EH_CONT:.*]]
// CHECK: [[EH_CONT]]:
// CHECK: [[CATCH]]:
// CHECK: catchret from [[CATCHPAD]] to label %[[CATCHRET_DEST:.*]]
// CHECK: [[CATCHRET_DEST]]:
// CHECK-NEXT: br label %[[EH_CONT]]

int explicitRethrow(void) {
  @try {
    mayThrow();
  } @catch (...) {
    @throw;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @explicitRethrow
// CHECK: invoke void @mayThrow()
// CHECK-NEXT: to label %{{.*}} unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind to caller
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr null]
// CHECK: br label %[[CATCH_ALL:.*]]
// CHECK: [[CATCH_ALL]]:
// CHECK: invoke void @__cxa_rethrow(){{.*}}[ "funclet"(token [[CATCHPAD]]) ]
// CHECK-NEXT: to label %[[UNREACHABLE:.*]] unwind label %{{.*}}
// CHECK: [[UNREACHABLE]]:
// CHECK-NEXT: unreachable
