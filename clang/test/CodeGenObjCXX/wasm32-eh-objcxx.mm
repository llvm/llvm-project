// REQUIRES: webassembly-registered-target
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
// CHECK: [[CLEANUPPAD:%.*]] = cleanuppad within none []
// CHECK: invoke{{.*}} @_ZN18ThrowingDestructorD1Ev{{.*}}[ "funclet"(token [[CLEANUPPAD]]) ]
// CHECK: cleanupret from [[CLEANUPPAD]] unwind label %{{.*}}
// CHECK: [[CATCHSWITCH:%.*]] = catchswitch within none [label %{{.*}}] unwind to caller
// CHECK: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr null]
// CHECK: invoke{{.*}} @_ZN18ThrowingDestructorD1Ev{{.*}}[ "funclet"(token [[CATCHPAD]]) ]
// CHECK: catchret from [[CATCHPAD]] to label %{{.*}}

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
// CHECK: [[CATCHSWITCH:%.*]] = catchswitch within none [label %{{.*}}] unwind label %{{.*}}
// CHECK: [[CATCHPAD:%.*]] = catchpad within [[CATCHSWITCH]] [ptr @__objc_eh_typeinfo_Object, ptr @_ZTIi]
// CHECK: invoke void @_Z{{[0-9]+}}mayThrowObjCv() [ "funclet"(token [[CATCHPAD]]) ]
// CHECK: [[NESTED_SWITCH:%.*]] = catchswitch within [[CATCHPAD]] [label %{{.*}}] unwind label %{{.*}}
// CHECK: [[NESTED_PAD:%.*]] = catchpad within [[NESTED_SWITCH]] [ptr @__objc_eh_typeinfo_Object]
// CHECK: invoke void @llvm.wasm.rethrow(){{.*}}[ "funclet"(token [[NESTED_PAD]]) ]
// CHECK: catchret from [[CATCHPAD]] to label %{{.*}}
// CHECK: invoke void @llvm.wasm.rethrow(){{.*}}[ "funclet"(token [[CATCHPAD]]) ]
// CHECK: [[OUTER_SWITCH:%.*]] = catchswitch within none [label %{{.*}}] unwind to caller
// CHECK: [[OUTER_PAD:%.*]] = catchpad within [[OUTER_SWITCH]] [ptr null]
