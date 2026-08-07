// RUN: %clang_cc1 -target-feature +exception-handling -triple wasm32-unknown-emscripten -fobjc-runtime=gnustep-2.2 -fexceptions -fobjc-exceptions -fcxx-exceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm -o - %s | FileCheck --enable-var-scope %s

struct ThrowingDestructor {
  ~ThrowingDestructor() noexcept(false);
};

extern void mayThrowCXX();

int cxxDestructorsAroundCatch() {
  try {
    ThrowingDestructor guard;
    mayThrowCXX();
  } catch (...) {
    ThrowingDestructor caught;
    return 1;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @_Z25cxxDestructorsAroundCatchv
// CHECK: invoke void @_Z{{[0-9]+}}mayThrowCXXv()
// CHECK-NEXT: to label %[[INVOKE_CONT:.*]] unwind label %[[EHCLEANUP:.*]]
// CHECK: [[INVOKE_CONT]]:
// CHECK: invoke{{.*}} @_ZN18ThrowingDestructorD1Ev
// CHECK-NEXT: to label %{{.*}} unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[EHCLEANUP]]:
// CHECK: [[CLEANUPPAD:%.*]] = cleanuppad within none []
// CHECK: invoke{{.*}} @_ZN18ThrowingDestructorD1Ev{{.*}}[ "funclet"(token [[CLEANUPPAD]]) ]
// CHECK: cleanupret from [[CLEANUPPAD]] unwind label %[[CATCH_DISPATCH]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind to caller
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr null]
// CHECK: br label %[[CATCH_ALL:.*]]
// CHECK: [[CATCH_ALL]]:
// CHECK: invoke{{.*}} @_ZN18ThrowingDestructorD1Ev{{.*}}[ "funclet"(token [[CATCHPAD]]) ]
// CHECK: catchret from [[CATCHPAD]] to label %{{.*}}
// CHECK: [[CLEANUPPAD1:%.*]] = cleanuppad within [[CATCHPAD]] []
// CHECK: cleanupret from [[CLEANUPPAD1]] unwind to caller

__attribute__((objc_root_class)) @interface Object
@end

extern void mayThrowObjC();

int combinedCxxObjcEH() {
  @try {
    try {
      mayThrowCXX();
    } catch (Object *exception) {
      @try {
        mayThrowObjC();
      } @catch (Object *nestedException) {
        return 1;
      }
      return 2;
    } catch (int value) {
      return value;
    }
  } @catch (...) {
    return 3;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @_Z{{[0-9]+}}combinedCxxObjcEHv
// CHECK: invoke void @_Z{{[0-9]+}}mayThrowCXXv()
// CHECK-NEXT: to label %{{.*}} unwind label %[[CATCH_DISPATCH:.*]]
// CHECK: [[CATCH_DISPATCH]]:
// CHECK-NEXT: [[CATCHSWITCH:%.*]] = catchswitch within none [label %[[CATCH_START:.*]]] unwind label %[[CATCH_DISPATCH1:.*]]
// CHECK: [[CATCH_START]]:
// CHECK-NEXT: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr @__objc_eh_typeinfo_Object, ptr @_ZTIi]
// CHECK: br i1 %{{.*}}, label %[[CATCH2:.*]], label %[[CATCH_FALLTHROUGH:.*]]
// CHECK: [[CATCH2]]:
// CHECK: invoke void @_Z{{[0-9]+}}mayThrowObjCv()
// CHECK-NEXT: to label %{{.*}} unwind label %[[CATCH_DISPATCH5:.*]]
// CHECK: [[CATCH_DISPATCH5]]:
// CHECK-NEXT: [[CATCHSWITCH1:%.*]] = catchswitch within [[CATCHPAD]] [label %[[CATCH_START6:.*]]] unwind label %[[EHCLEANUP:.*]]
// CHECK: [[CATCH_START6]]:
// CHECK-NEXT: [[CATCHPAD1:%.*]] = catchpad within [[CATCHSWITCH1]] [ptr @__objc_eh_typeinfo_Object]
// CHECK: br i1 %{{.*}}, label %[[CATCH9:.*]], label %[[RETHROW8:.*]]
// CHECK: [[RETHROW8]]:
// CHECK: invoke void @llvm.wasm.rethrow(){{.*}}[ "funclet"(token [[CATCHPAD1]]) ]
// CHECK: [[CATCH_FALLTHROUGH]]:
// CHECK: br i1 %{{.*}}, label %[[CATCH:.*]], label %[[RETHROW:.*]]
// CHECK: [[CATCH]]:
// CHECK: catchret from [[CATCHPAD]] to label %{{.*}}
// CHECK: [[RETHROW]]:
// CHECK: invoke void @llvm.wasm.rethrow(){{.*}}[ "funclet"(token [[CATCHPAD]]) ]
// CHECK-NEXT: to label %{{.*}} unwind label %[[CATCH_DISPATCH1]]
// CHECK: [[CATCH_DISPATCH1]]:
// CHECK-NEXT: [[CATCHSWITCH2:%.*]] = catchswitch within none [label %[[CATCH_START15:.*]]] unwind to caller
// CHECK: [[CATCH_START15]]:
// CHECK-NEXT: [[CATCHPAD2:%.*]] = catchpad within [[CATCHSWITCH2]] [ptr null]
// CHECK: br label %[[CATCH_ALL:.*]]
// CHECK: [[CATCH9]]:
// CHECK: catchret from [[CATCHPAD1]] to label %{{.*}}
// CHECK: [[EHCLEANUP]]:
// CHECK: [[CLEANUPPAD:%.*]] = cleanuppad within [[CATCHPAD]] []
// CHECK: cleanupret from [[CLEANUPPAD]] unwind label %[[CATCH_DISPATCH1]]
// CHECK: [[CATCH_ALL]]:
// CHECK: catchret from [[CATCHPAD2]] to label %{{.*}}
