// RUN: %clang_cc1 -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -fexceptions -fobjc-exceptions -O2 -o - %s | FileCheck %s

// Test we maintain at least a basic amount of interoperation between
// ObjC and C++ exceptions in the legacy runtime.

// rdar://12364847

void foo(void);

void test0(id obj) {
  @synchronized(obj) {
    foo();
  }
}
// CHECK-LABEL:    define void @_Z5test0P11objc_object(
//   Enter the @synchronized block.
// CHECK:      call i32 @objc_sync_enter(i8* [[OBJ:%.*]])
// CHECK:      call void @objc_exception_try_enter([[BUF_T:%.*]]* nonnull [[BUF:%.*]])
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BUF_T]], [[BUF_T]]* [[BUF]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = call i32 @_setjmp(i32* nonnull [[T0]])
// CHECK-NEXT: [[T2:%.*]] = icmp eq i32 [[T1]], 0
// CHECK-NEXT: br i1 [[T2]],

//   Body.
// CHECK:      invoke void @_Z3foov()

//   Leave the @synchronized.  The reload of obj here is unnecessary.
// CHECK:      call void @objc_exception_try_exit([[BUF_T]]* nonnull [[BUF]])
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8**
// CHECK-NEXT: call i32 @objc_sync_exit(i8* [[T0]])
// CHECK-NEXT: ret void

//   Real EH cleanup.
// CHECK:      [[T0:%.*]] = landingpad
// CHECK-NEXT:    cleanup
// CHECK-NEXT: call void @objc_exception_try_exit([[BUF_T]]* nonnull [[BUF]])
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8**
// CHECK-NEXT: call i32 @objc_sync_exit(i8* [[T0]])
// CHECK-NEXT: resume

//   ObjC EH "cleanup".
// CHECK:      [[T0:%.*]] = load i8*, i8**
// CHECK-NEXT: call i32 @objc_sync_exit(i8* [[T0]])
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_exception_extract([[BUF_T]]* nonnull [[BUF]])
// CHECK-NEXT: call void @objc_exception_throw(i8* [[T0]])
// CHECK-NEXT: unreachable

void test1(id obj, bool *failed) {
  @try {
    foo();
  } @catch (...) {
    *failed = true;
  }
}
// CHECK-LABEL:    define void @_Z5test1P11objc_objectPb(
//   Enter the @try block.
// CHECK:      call void @objc_exception_try_enter([[BUF_T]]* nonnull [[BUF:%.*]])
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BUF_T]], [[BUF_T]]* [[BUF]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = call i32 @_setjmp(i32* nonnull [[T0]])
// CHECK-NEXT: [[T2:%.*]] = icmp eq i32 [[T1]], 0
// CHECK-NEXT: br i1 [[T2]],

//   Body.
// CHECK:      invoke void @_Z3foov()

//   Catch handler.  Reload of 'failed' address is unnecessary.
// CHECK:      [[T0:%.*]] = load i8*, i8**
// CHECK-NEXT: store i8 1, i8* [[T0]],
// CHECK-NEXT: br label

//   Leave the @try.
// CHECK:      call void @objc_exception_try_exit([[BUF_T]]* nonnull [[BUF]])
// CHECK-NEXT: br label
// CHECK:      ret void


//   Real EH cleanup.
// CHECK:      [[T0:%.*]] = landingpad
// CHECK-NEXT:    cleanup
// CHECK-NEXT: call void @objc_exception_try_exit([[BUF_T]]* nonnull [[BUF]])
// CHECK-NEXT: resume

