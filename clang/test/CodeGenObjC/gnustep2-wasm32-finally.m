// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -target-feature +exception-handling -triple wasm32-unknown-emscripten -fobjc-runtime=gnustep-2.2 -fexceptions -fobjc-exceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm -o - %s | FileCheck %s

__attribute__((objc_root_class)) @interface Object @end
extern void mayThrowObjC();

void emptyFinally(void) {
  @try {
    mayThrowObjC();
  } @finally {
  }
}

// CHECK-LABEL: define{{.*}} @emptyFinally
// CHECK:       catch.dispatch:
// CHECK-NEXT:  [[EMPTY_SWITCH:%.*]] = catchswitch within none [label %catch.start] unwind to caller
// CHECK:       catch.start:
// CHECK-NEXT:  [[EMPTY_PAD:%.*]] = catchpad within [[EMPTY_SWITCH]] [ptr null]
// CHECK:       br label %finally.catchall
// CHECK:       {{^}}cleanup:
// CHECK:       %finally.shouldthrow = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK:       finally.rethrow:
// CHECK-NEXT:  invoke void @__cxa_rethrow()
// CHECK:       finally.cont:
// CHECK:       finally.catchall:
// CHECK-NEXT:  %exn = load ptr, ptr %exn.slot
// CHECK-NEXT:  %{{.*}} = call ptr @__cxa_begin_catch(ptr %exn)
// CHECK-NEXT:  store i1 true, ptr %finally.for-eh
// CHECK-NEXT:  store i32 2, ptr %cleanup.dest.slot
// CHECK-NEXT:  catchret from [[EMPTY_PAD]] to label %{{.*}}
// CHECK:       ehcleanup:
// CHECK-NEXT:  [[EMPTY_CLEANUP:%.*]] = cleanuppad within none []
// CHECK-NEXT:  %finally.endcatch = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.endcatch, label %{{.*}}, label %finally.cleanup.cont
// CHECK:       finally.cleanup.cont:
// CHECK-NEXT:  cleanupret from [[EMPTY_CLEANUP]] unwind to caller

int finallySimple(Object *object) {
  int value = 0;
  @try {
    mayThrowObjC();
    value = 1;
  } @catch (...) {
    value = 2;
  } @finally {
    value += object != (Object *)0;
  }
  return value;
}

// CHECK-LABEL: define{{.*}} @finallySimple
// CHECK:       invoke void @mayThrowObjC()
// CHECK-NEXT:          to label %invoke.cont unwind label %catch.dispatch
// CHECK:       catch.dispatch:
// CHECK-NEXT:  [[SIMPLE_SWITCH:%.*]] = catchswitch within none [label %catch.start] unwind label %catch.dispatch2
// CHECK:       catch.start:
// CHECK-NEXT:  [[SIMPLE_PAD:%.*]] = catchpad within [[SIMPLE_SWITCH]] [ptr null]
// CHECK:       br label %catch
// CHECK:       invoke.cont:
// CHECK-NEXT:  store i32 1, ptr %value
// CHECK-NEXT:  store i32 0, ptr %cleanup.dest.slot
// CHECK-NEXT:  br label %cleanup
// CHECK:       cleanup:
// CHECK:       %add = add nsw i32
// CHECK-NEXT:  store i32 %add, ptr %value
// CHECK-NEXT:  %finally.shouldthrow = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK:       finally.rethrow:
// CHECK-NEXT:  invoke void @__cxa_rethrow()
// CHECK-NEXT:          to label %unreachable unwind label %ehcleanup
// CHECK:       finally.cont:
// CHECK-NEXT:  store i32 %cleanup.dest.saved, ptr %cleanup.dest.slot
// CHECK-NEXT:  %cleanup.dest = load i32, ptr %cleanup.dest.slot
// CHECK-NEXT:  switch i32 %cleanup.dest, label %unreachable [
// CHECK-NEXT:    i32 0, label %cleanup.cont
// CHECK-NEXT:    i32 2, label %eh.cont
// CHECK-NEXT:    i32 3, label %unreachable
// CHECK-NEXT:  ]
// CHECK:       cleanup.cont:
// CHECK-NEXT:  br label %eh.cont
// CHECK:       eh.cont:
// CHECK-NEXT:  %{{.*}} = load i32, ptr %value
// CHECK-NEXT:  ret i32 %{{.*}}
// CHECK:       catch:
// CHECK-NEXT:  %exn = load ptr, ptr %exn.slot
// CHECK-NEXT:  %exn.adjusted = call ptr @__cxa_begin_catch(ptr %exn)
// CHECK-NEXT:  store i32 2, ptr %value
// CHECK-NEXT:  invoke void @__cxa_end_catch()
// CHECK-NEXT:          to label %invoke.cont1 unwind label %catch.dispatch2
// CHECK:       catch.dispatch2:
// CHECK-NEXT:  [[SIMPLE_FINALLY_SWITCH:%.*]] = catchswitch within none [label %catch.start3] unwind to caller
// CHECK:       catch.start3:
// CHECK-NEXT:  [[SIMPLE_FINALLY_PAD:%.*]] = catchpad within [[SIMPLE_FINALLY_SWITCH]] [ptr null]
// CHECK:       br label %finally.catchall
// CHECK:       invoke.cont1:
// CHECK-NEXT:  catchret from [[SIMPLE_PAD]] to label %catchret.dest
// CHECK:       catchret.dest:
// CHECK-NEXT:  store i32 2, ptr %cleanup.dest.slot
// CHECK-NEXT:  br label %cleanup
// CHECK:       finally.catchall:
// CHECK-NEXT:  %exn4 = load ptr, ptr %exn.slot
// CHECK-NEXT:  %{{.*}} = call ptr @__cxa_begin_catch(ptr %exn4)
// CHECK-NEXT:  store i1 true, ptr %finally.for-eh
// CHECK-NEXT:  store i32 3, ptr %cleanup.dest.slot
// CHECK-NEXT:  catchret from [[SIMPLE_FINALLY_PAD]] to label %catchret.dest5
// CHECK:       catchret.dest5:
// CHECK-NEXT:  br label %cleanup
// CHECK:       ehcleanup:
// CHECK-NEXT:  [[SIMPLE_CLEANUP:%.*]] = cleanuppad within none []
// CHECK-NEXT:  %finally.endcatch = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.endcatch, label %finally.endcatch6, label %finally.cleanup.cont
// CHECK:       finally.endcatch6:
// CHECK-NEXT:  invoke void @__cxa_end_catch()
// CHECK-NEXT:          to label %invoke.cont7 unwind label %terminate
// CHECK:       invoke.cont7:
// CHECK-NEXT:  br label %finally.cleanup.cont
// CHECK:       finally.cleanup.cont:
// CHECK-NEXT:  cleanupret from [[SIMPLE_CLEANUP]] unwind to caller

int finallyNoCatch(Object *object) {
  int value = 0;
  @try {
    mayThrowObjC();
    value = 1;
  } @finally {
    value += object != (Object *)0;
  }
  return value;
}

// CHECK-LABEL: define{{.*}} @finallyNoCatch
// CHECK:       catch.dispatch:
// CHECK-NEXT:  [[NO_CATCH_SWITCH:%.*]] = catchswitch within none [label %catch.start] unwind to caller
// CHECK:       catch.start:
// CHECK-NEXT:  [[NO_CATCH_PAD:%.*]] = catchpad within [[NO_CATCH_SWITCH]] [ptr null]
// CHECK:       br label %finally.catchall
// CHECK:       {{^}}cleanup:
// CHECK:       %finally.shouldthrow = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK:       finally.rethrow:
// CHECK-NEXT:  invoke void @__cxa_rethrow()
// CHECK:       finally.cont:
// CHECK:       finally.catchall:
// CHECK-NEXT:  %exn = load ptr, ptr %exn.slot
// CHECK-NEXT:  %{{.*}} = call ptr @__cxa_begin_catch(ptr %exn)
// CHECK-NEXT:  store i1 true, ptr %finally.for-eh
// CHECK-NEXT:  store i32 2, ptr %cleanup.dest.slot
// CHECK-NEXT:  catchret from [[NO_CATCH_PAD]] to label %{{.*}}
// CHECK:       ehcleanup:
// CHECK-NEXT:  [[NO_CATCH_CLEANUP:%.*]] = cleanuppad within none []
// CHECK-NEXT:  %finally.endcatch = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.endcatch, label %{{.*}}, label %finally.cleanup.cont
// CHECK:       finally.cleanup.cont:
// CHECK-NEXT:  cleanupret from [[NO_CATCH_CLEANUP]] unwind to caller

int throwInCatchFinally(Object *object) {
  @try {
    mayThrowObjC();
  } @catch (...) {
    @throw;
  } @finally {
    (void)object;
  }
}

// CHECK-LABEL: define{{.*}} @throwInCatchFinally
// CHECK:       {{^}}cleanup:
// CHECK:       %finally.shouldthrow = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK:       finally.rethrow:
// CHECK-NEXT:  invoke void @__cxa_rethrow()
// CHECK:       finally.cont:
// CHECK:       catch:
// CHECK:       invoke void @__cxa_rethrow()
// CHECK-NEXT:          to label %unreachable unwind label %ehcleanup
// CHECK:       catch.dispatch{{[0-9]+}}:
// CHECK-NEXT:  [[CATCH_FINALLY_SWITCH:%.*]] = catchswitch within none [label %catch.start{{[0-9]+}}] unwind to caller
// CHECK:       catch.start{{[0-9]+}}:
// CHECK:       [[CATCH_FINALLY_PAD:%.*]] = catchpad within [[CATCH_FINALLY_SWITCH]] [ptr null]
// CHECK:       br label %finally.catchall
// CHECK:       finally.catchall:
// CHECK:       %{{.*}} = call ptr @__cxa_begin_catch(ptr %{{.*}})
// CHECK-NEXT:  store i1 true, ptr %finally.for-eh
// CHECK-NEXT:  store i32 3, ptr %cleanup.dest.slot
// CHECK-NEXT:  catchret from [[CATCH_FINALLY_PAD]] to label %{{.*}}
// CHECK:       ehcleanup{{[0-9]+}}:
// CHECK-NEXT:  [[CATCH_FINALLY_CLEANUP:%.*]] = cleanuppad within none []
// CHECK:       finally.cleanup.cont:
// CHECK-NEXT:  cleanupret from [[CATCH_FINALLY_CLEANUP]] unwind to caller

int throwInFinally(Object *object) {
  @try {
    mayThrowObjC();
  } @finally {
    @throw object;
  }
}

// CHECK-LABEL: define{{.*}} @throwInFinally
// CHECK:       cleanup:
// CHECK:       invoke void @objc_exception_throw(ptr %{{.*}})
// CHECK-NEXT:          to label %invoke.cont1 unwind label %ehcleanup
// CHECK:       invoke.cont1:
// CHECK-NEXT:  unreachable
// CHECK:       finally.catchall:
// CHECK:       catchret from %{{.*}} to label %{{.*}}
// CHECK:       ehcleanup:
// CHECK:       [[THROW_CLEANUP:%.*]] = cleanuppad within none []
// CHECK:       finally.cleanup.cont:
// CHECK-NEXT:  cleanupret from [[THROW_CLEANUP]] unwind to caller
// CHECK-NOT:  finally.rethrow:

int throwInFinallyNoException(Object *object) {
  @try {
  } @finally {
    @throw object;
  }
}

// CHECK-LABEL: define{{.*}} @throwInFinallyNoException
// CHECK:       entry:
// CHECK:       invoke void @objc_exception_throw(ptr %{{.*}})
// CHECK-NEXT:          to label %invoke.cont unwind label %ehcleanup
// CHECK:       invoke.cont:
// CHECK-NEXT:  unreachable
// CHECK:       ehcleanup:
// CHECK-NEXT:  [[NO_EXCEPTION_CLEANUP:%.*]] = cleanuppad within none []
// CHECK-NEXT:  %finally.endcatch = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.endcatch, label %{{.*}}, label %finally.cleanup.cont
// CHECK:       finally.cleanup.cont:
// CHECK-NEXT:  cleanupret from [[NO_EXCEPTION_CLEANUP]] unwind to caller
// CHECK-NOT:  catchswitch within none
// CHECK-NOT:  finally.rethrow:

int nestedTryCatchFinally(Object *object) {
  int value = 0;
  @try {
    @try {
      mayThrowObjC();
    } @catch (...) {
      value = 1;
    } @finally {
      value += 2;
    }
  } @catch (...) {
    value = 3;
  } @finally {
    value += object != (Object *)0;
  }
  return value;
}

// CHECK-LABEL: define{{.*}} @nestedTryCatchFinally
// CHECK:       {{^}}cleanup:
// CHECK:       %finally.shouldthrow = load i1, ptr %finally.for-eh1
// CHECK-NEXT:  br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK:       finally.rethrow:
// CHECK-NEXT:  invoke void @__cxa_rethrow()
// CHECK:       {{^}}cleanup{{[0-9]+}}:
// CHECK:       %finally.shouldthrow{{[0-9]+}} = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.shouldthrow{{[0-9]+}}, label %finally.rethrow{{[0-9]+}}, label %finally.cont{{[0-9]+}}
// CHECK:       finally.rethrow{{[0-9]+}}:
// CHECK-NEXT:  invoke void @__cxa_rethrow()
// CHECK:       finally.catchall:
// CHECK:       catchret from %{{.*}} to label %{{.*}}
// CHECK:       finally.catchall{{[0-9]+}}:
// CHECK:       catchret from %{{.*}} to label %{{.*}}

int gotoOutFinally(Object *object) {
  int value = 0;
  @try {
    value = 1;
    goto done;
  } @finally {
    value += object != (Object *)0;
  }
done:
  return value;
}

// CHECK-LABEL: define{{.*}} @gotoOutFinally
// CHECK:       entry:
// CHECK:       store i32 1, ptr %value
// CHECK-NEXT:  store i32 3, ptr %cleanup.dest.slot
// CHECK:       %finally.shouldthrow = load i1, ptr %finally.for-eh
// CHECK-NEXT:  br i1 %finally.shouldthrow, label %finally.rethrow, label %finally.cont
// CHECK:       finally.rethrow:
// CHECK-NEXT:  invoke void @__cxa_rethrow()
// CHECK:       finally.cont:
// CHECK:       i32 3, label %done
// CHECK:       ehcleanup:
// CHECK:       finally.cleanup.cont:
// CHECK-NEXT:  cleanupret from %{{.*}} unwind to caller
