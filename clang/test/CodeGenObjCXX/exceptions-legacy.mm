// RUN: %clang_cc1 -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -fexceptions -fobjc-exceptions -O2 -o - %s | FileCheck %s

// Test we maintain at least a basic amount of interoperation between
// ObjC and C++ exceptions in the legacy runtime.

void foo(void);

void test0(id obj) {
  @synchronized(obj) {
    foo();
  }
}
// CHECK-LABEL:    define{{.*}} void @_Z5test0P11objc_object(
//   Enter the @synchronized block.
// CHECK:      call i32 @objc_sync_enter(ptr [[OBJ:%.*]])
// CHECK:      call void @objc_exception_try_enter(ptr nonnull [[BUF:%.*]])
// CHECK-NEXT: [[T1:%.*]] = call i32 @_setjmp(ptr nonnull [[BUF]])
// CHECK-NEXT: [[T2:%.*]] = icmp eq i32 [[T1]], 0
// CHECK-NEXT: br i1 [[T2]],

//   Body.
// CHECK:      invoke void @_Z3foov()

//   Leave the @synchronized.  The reload of obj here is unnecessary.
// CHECK:      call void @objc_exception_try_exit(ptr nonnull [[BUF]])
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
// CHECK-NEXT: call i32 @objc_sync_exit(ptr [[T0]])
// CHECK-NEXT: ret void

//   Real EH cleanup.
// CHECK:      [[T0:%.*]] = landingpad
// CHECK-NEXT:    cleanup
// CHECK-NEXT: call void @objc_exception_try_exit(ptr nonnull [[BUF]])
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr
// CHECK-NEXT: call i32 @objc_sync_exit(ptr [[T0]])
// CHECK-NEXT: resume

//   ObjC EH "cleanup".
// CHECK:      [[T0:%.*]] = load ptr, ptr
// CHECK-NEXT: call i32 @objc_sync_exit(ptr [[T0]])
// CHECK-NEXT: [[T0:%.*]] = call ptr @objc_exception_extract(ptr nonnull [[BUF]])
// CHECK-NEXT: call void @objc_exception_throw(ptr [[T0]])
// CHECK-NEXT: unreachable

void test1(id obj, bool *failed) {
  @try {
    foo();
  } @catch (...) {
    *failed = true;
  }
}
// CHECK-LABEL:    define{{.*}} void @_Z5test1P11objc_objectPb(
//   Enter the @try block.
// CHECK:      call void @objc_exception_try_enter(ptr nonnull [[BUF:%.*]])
// CHECK-NEXT: [[T1:%.*]] = call i32 @_setjmp(ptr nonnull [[BUF]])
// CHECK-NEXT: [[T2:%.*]] = icmp eq i32 [[T1]], 0
// CHECK-NEXT: br i1 [[T2]],

//   Body.
// CHECK:      invoke void @_Z3foov()

//   Catch handler.  Reload of 'failed' address is unnecessary.
// CHECK:      [[T0:%.*]] = load ptr, ptr
// CHECK-NEXT: store i8 1, ptr [[T0]],
// CHECK-NEXT: br label

//   Leave the @try.
// CHECK:      call void @objc_exception_try_exit(ptr nonnull [[BUF]])
// CHECK-NEXT: br label
// CHECK:      ret void


//   Real EH cleanup.
// CHECK:      [[T0:%.*]] = landingpad
// CHECK-NEXT:    cleanup
// CHECK-NEXT: call void @objc_exception_try_exit(ptr nonnull [[BUF]])
// CHECK-NEXT: resume

