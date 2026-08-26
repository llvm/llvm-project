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

int cleanupInTryFinally() {
  @try {
    ThrowingDestructor object;
    mayThrowObjC();
  } @finally {
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @_Z{{[0-9]+}}cleanupInTryFinallyv
// CHECK: invoke void @_Z{{[0-9]+}}mayThrowObjCv()
// CHECK-NEXT:          to label %invoke.cont unwind label %ehcleanup
// CHECK: invoke.cont:
// CHECK-NEXT: %{{.*}} = invoke noundef ptr @_ZN18ThrowingDestructorD1Ev{{.*}}%object
// CHECK-NEXT:          to label %invoke.cont1 unwind label %catch.dispatch
// CHECK: invoke.cont1:
// CHECK-NEXT: store i32 0, ptr %cleanup.dest.slot
// CHECK-NEXT: br label %cleanup
// CHECK: cleanup:
// CHECK-NEXT: %cleanup.dest.saved = load i32, ptr %cleanup.dest.slot
// CHECK-NEXT: %finally.shouldthrow = load i1, ptr %finally.for-eh
// CHECK-NEXT: br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK: finally.rethrow:
// CHECK-NEXT: invoke void @__cxa_rethrow()
// CHECK-NEXT:          to label %unreachable unwind label %ehcleanup4
// CHECK: finally.cont:
// CHECK-NEXT: store i32 %cleanup.dest.saved, ptr %cleanup.dest.slot
// CHECK-NEXT: %cleanup.dest = load i32, ptr %cleanup.dest.slot
// CHECK-NEXT: switch i32 %cleanup.dest, label %unreachable [
// CHECK-NEXT: i32 0, label %cleanup.cont
// CHECK-NEXT: i32 2, label %unreachable
// CHECK-NEXT: ]
// CHECK: cleanup.cont:
// CHECK-NEXT: ret i32 0
// CHECK: ehcleanup:
// CHECK-NEXT: [[TRY_CLEANUP:%.*]] = cleanuppad within none []
// CHECK-NEXT: %{{.*}} = invoke noundef ptr @_ZN18ThrowingDestructorD1Ev{{.*}}[ "funclet"(token [[TRY_CLEANUP]]) ]
// CHECK-NEXT:          to label %invoke.cont2 unwind label %terminate
// CHECK: invoke.cont2:
// CHECK-NEXT: cleanupret from [[TRY_CLEANUP]] unwind label %catch.dispatch
// CHECK: catch.dispatch:
// CHECK-NEXT: [[TRY_SWITCH:%.*]] = catchswitch within none [label %catch.start] unwind to caller
// CHECK: catch.start:
// CHECK-NEXT: [[TRY_PAD:%.*]] = catchpad within [[TRY_SWITCH]] [ptr null]
// CHECK: br label %finally.catchall
// CHECK: finally.catchall:
// CHECK-NEXT: %exn = load ptr, ptr %exn.slot
// CHECK-NEXT: %{{.*}} = call ptr @__cxa_begin_catch(ptr %exn)
// CHECK-NEXT: store i1 true, ptr %finally.for-eh
// CHECK-NEXT: store i32 2, ptr %cleanup.dest.slot
// CHECK-NEXT: catchret from [[TRY_PAD]] to label %catchret.dest
// CHECK: catchret.dest:
// CHECK-NEXT: br label %cleanup
// CHECK: ehcleanup4:
// CHECK-NEXT: [[TRY_FINALLY_CLEANUP:%.*]] = cleanuppad within none []
// CHECK-NEXT: %finally.endcatch = load i1, ptr %finally.for-eh
// CHECK-NEXT: br i1 %finally.endcatch, label %finally.endcatch5, label %finally.cleanup.cont
// CHECK: finally.endcatch5:
// CHECK-NEXT: invoke void @__cxa_end_catch()
// CHECK-NEXT:          to label %invoke.cont6 unwind label %terminate7
// CHECK: invoke.cont6:
// CHECK-NEXT: br label %finally.cleanup.cont
// CHECK: finally.cleanup.cont:
// CHECK-NEXT: cleanupret from [[TRY_FINALLY_CLEANUP]] unwind to caller

int cleanupInCatchFinally() {
  @try {
    mayThrowObjC();
  } @catch (...) {
    ThrowingDestructor object;
    return 1;
  } @finally {
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @_Z{{[0-9]+}}cleanupInCatchFinallyv
// CHECK: invoke void @_Z{{[0-9]+}}mayThrowObjCv()
// CHECK-NEXT:          to label %invoke.cont unwind label %catch.dispatch
// CHECK: catch.dispatch:
// CHECK-NEXT: [[CATCH_SWITCH:%.*]] = catchswitch within none [label %catch.start] unwind label %catch.dispatch4
// CHECK: catch.start:
// CHECK-NEXT: [[CATCH_PAD:%.*]] = catchpad within [[CATCH_SWITCH]] [ptr null]
// CHECK: br label %catch
// CHECK: invoke.cont:
// CHECK-NEXT: store i32 0, ptr %cleanup.dest.slot
// CHECK-NEXT: br label %cleanup
// CHECK: cleanup:
// CHECK-NEXT: %cleanup.dest.saved = load i32, ptr %cleanup.dest.slot
// CHECK-NEXT: %finally.shouldthrow = load i1, ptr %finally.for-eh
// CHECK-NEXT: br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK: finally.rethrow:
// CHECK-NEXT: invoke void @__cxa_rethrow()
// CHECK-NEXT:          to label %unreachable unwind label %ehcleanup8
// CHECK: finally.cont:
// CHECK-NEXT: store i32 %cleanup.dest.saved, ptr %cleanup.dest.slot
// CHECK-NEXT: %cleanup.dest = load i32, ptr %cleanup.dest.slot
// CHECK-NEXT: switch i32 %cleanup.dest, label %unreachable [
// CHECK-NEXT: i32 0, label %cleanup.cont
// CHECK-NEXT: i32 1, label %return
// CHECK-NEXT: i32 3, label %unreachable
// CHECK-NEXT: ]
// CHECK: cleanup.cont:
// CHECK-NEXT: br label %eh.cont
// CHECK: eh.cont:
// CHECK-NEXT: store i32 0, ptr %retval
// CHECK-NEXT: br label %return
// CHECK: catch:
// CHECK-NEXT: %exn = load ptr, ptr %exn.slot
// CHECK-NEXT: %exn.adjusted = call ptr @__cxa_begin_catch(ptr %exn)
// CHECK-NEXT: store i32 1, ptr %retval
// CHECK-NEXT: store i32 1, ptr %cleanup.dest.slot
// CHECK-NEXT: %{{.*}} = invoke noundef ptr @_ZN18ThrowingDestructorD1Ev{{.*}}[ "funclet"(token [[CATCH_PAD]]) ]
// CHECK-NEXT:          to label %invoke.cont1 unwind label %ehcleanup
// CHECK: invoke.cont1:
// CHECK-NEXT: invoke void @__cxa_end_catch()
// CHECK-NEXT:          to label %invoke.cont2 unwind label %catch.dispatch4
// CHECK: invoke.cont2:
// CHECK-NEXT: catchret from [[CATCH_PAD]] to label %catchret.dest
// CHECK: ehcleanup:
// CHECK-NEXT: [[CATCH_CLEANUP:%.*]] = cleanuppad within [[CATCH_PAD]] []
// CHECK-NEXT: invoke void @__cxa_end_catch()
// CHECK-NEXT:          to label %invoke.cont3 unwind label %terminate
// CHECK: invoke.cont3:
// CHECK-NEXT: cleanupret from [[CATCH_CLEANUP]] unwind label %catch.dispatch4
// CHECK: catch.dispatch4:
// CHECK-NEXT: [[FINALLY_SWITCH:%.*]] = catchswitch within none [label %catch.start5] unwind to caller
// CHECK: catch.start5:
// CHECK-NEXT: [[FINALLY_PAD:%.*]] = catchpad within [[FINALLY_SWITCH]] [ptr null]
// CHECK: br label %finally.catchall
// CHECK: catchret.dest:
// CHECK-NEXT: br label %cleanup
// CHECK: finally.catchall:
// CHECK-NEXT: %exn6 = load ptr, ptr %exn.slot
// CHECK-NEXT: %{{.*}} = call ptr @__cxa_begin_catch(ptr %exn6)
// CHECK-NEXT: store i1 true, ptr %finally.for-eh
// CHECK-NEXT: store i32 3, ptr %cleanup.dest.slot
// CHECK-NEXT: catchret from [[FINALLY_PAD]] to label %catchret.dest7
// CHECK: catchret.dest7:
// CHECK-NEXT: br label %cleanup
// CHECK: ehcleanup8:
// CHECK-NEXT: [[CATCH_FINALLY_CLEANUP:%.*]] = cleanuppad within none []
// CHECK-NEXT: %finally.endcatch = load i1, ptr %finally.for-eh
// CHECK-NEXT: br i1 %finally.endcatch, label %finally.endcatch9, label %finally.cleanup.cont
// CHECK: finally.endcatch9:
// CHECK-NEXT: invoke void @__cxa_end_catch()
// CHECK-NEXT:          to label %invoke.cont10 unwind label %terminate11
// CHECK: invoke.cont10:
// CHECK-NEXT: br label %finally.cleanup.cont
// CHECK: finally.cleanup.cont:
// CHECK-NEXT: cleanupret from [[CATCH_FINALLY_CLEANUP]] unwind to caller
// CHECK: return:
// CHECK-NEXT: %{{.*}} = load i32, ptr %retval
// CHECK-NEXT: ret i32 %{{.*}}

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
