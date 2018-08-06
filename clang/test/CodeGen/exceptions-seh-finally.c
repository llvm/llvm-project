// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

void abort(void) __attribute__((noreturn));
void might_crash(void);
void cleanup(void);
int check_condition(void);
void basic_finally(void) {
  __try {
    might_crash();
  } __finally {
    cleanup();
  }
}

// CHECK-LABEL: define dso_local void @basic_finally()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"?fin$0@0@basic_finally@@"({{i8( zeroext)?}} 0, i8* %[[fp]])
// CHECK-NEXT: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"?fin$0@0@basic_finally@@"({{i8( zeroext)?}} 1, i8* %[[fp]])
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK: define internal void @"?fin$0@0@basic_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs:#[0-9]+]]
// CHECK: call void @cleanup()

// Mostly check that we don't double emit 'r' which would crash.
void decl_in_finally(void) {
  __try {
    might_crash();
  } __finally {
    int r;
  }
}

// Ditto, don't crash double emitting 'l'.
void label_in_finally(void) {
  __try {
    might_crash();
  } __finally {
l:
    cleanup();
    if (check_condition())
      goto l;
  }
}

// CHECK-LABEL: define dso_local void @label_in_finally()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"?fin$0@0@label_in_finally@@"({{i8( zeroext)?}} 0, i8* %[[fp]])
// CHECK: ret void

// CHECK: define internal void @"?fin$0@0@label_in_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br label %[[l:[^ ]*]]
//
// CHECK: [[l]]
// CHECK: call void @cleanup()
// CHECK: call i32 @check_condition()
// CHECK: br i1 {{.*}}, label
// CHECK: br label %[[l]]

int crashed;
void use_abnormal_termination(void) {
  __try {
    might_crash();
  } __finally {
    crashed = __abnormal_termination();
  }
}

// CHECK-LABEL: define dso_local void @use_abnormal_termination()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"?fin$0@0@use_abnormal_termination@@"({{i8( zeroext)?}} 0, i8* %[[fp]])
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"?fin$0@0@use_abnormal_termination@@"({{i8( zeroext)?}} 1, i8* %[[fp]])
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK: define internal void @"?fin$0@0@use_abnormal_termination@@"({{i8( zeroext)?}} %[[abnormal:abnormal_termination]], i8* %frame_pointer)
// CHECK-SAME: [[finally_attrs]]
// CHECK: %[[abnormal_zext:[^ ]*]] = zext i8 %[[abnormal]] to i32
// CHECK: store i32 %[[abnormal_zext]], i32* @crashed
// CHECK-NEXT: ret void

void noreturn_noop_finally() {
  __try {
    __noop();
  } __finally {
    abort();
  }
}

// CHECK-LABEL: define dso_local void @noreturn_noop_finally()
// CHECK: call void @"?fin$0@0@noreturn_noop_finally@@"({{.*}})
// CHECK: ret void

// CHECK: define internal void @"?fin$0@0@noreturn_noop_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: call void @abort()
// CHECK: unreachable

void noreturn_finally() {
  __try {
    might_crash();
  } __finally {
    abort();
  }
}

// CHECK-LABEL: define dso_local void @noreturn_finally()
// CHECK: invoke void @might_crash()
// CHECK:     to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: call void @"?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: call void @"?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK: define internal void @"?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: call void @abort()
// CHECK: unreachable

int finally_with_return() {
  __try {
    return 42;
  } __finally {
  }
}
// CHECK-LABEL: define dso_local i32 @finally_with_return()
// CHECK: call void @"?fin$0@0@finally_with_return@@"({{.*}})
// CHECK-NEXT: ret i32 42

// CHECK: define internal void @"?fin$0@0@finally_with_return@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK-NOT: br i1
// CHECK-NOT: br label
// CHECK: ret void

int nested___finally___finally() {
  __try {
    __try {
    } __finally {
      return 1;
    }
  } __finally {
    // Intentionally no return here.
  }
  return 0;
}

// CHECK-LABEL: define dso_local i32 @nested___finally___finally
// CHECK: invoke void @"?fin$1@0@nested___finally___finally@@"({{.*}})
// CHECK:          to label %[[outercont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[outercont]]
// CHECK: call void @"?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK-NEXT: ret i32 0
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: call void @"?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK-LABEL: define internal void @"?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: ret void

// CHECK-LABEL: define internal void @"?fin$1@0@nested___finally___finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: unreachable

// FIXME: Our behavior seems suspiciously different.

int nested___finally___finally_with_eh_edge() {
  __try {
    __try {
      might_crash();
    } __finally {
      return 899;
    }
  } __finally {
    // Intentionally no return here.
  }
  return 912;
}
// CHECK-LABEL: define dso_local i32 @nested___finally___finally_with_eh_edge
// CHECK: invoke void @might_crash()
// CHECK-NEXT: to label %[[invokecont:[^ ]*]] unwind label %[[lpad1:[^ ]*]]
//
// [[invokecont]]
// CHECK: invoke void @"?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT:       to label %[[outercont:[^ ]*]] unwind label %[[lpad2:[^ ]*]]
//
// CHECK: [[outercont]]
// CHECK: call void @"?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT: ret i32 912
//
// CHECK: [[lpad1]]
// CHECK-NEXT: %[[innerpad:[^ ]*]] = cleanuppad
// CHECK: invoke void @"?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT:    label %[[innercleanupretbb:[^ ]*]] unwind label %[[lpad2:[^ ]*]]
//
// CHECK: [[innercleanupretbb]]
// CHECK-NEXT: cleanupret from %[[innerpad]] unwind label %[[lpad2]]
//
// CHECK: [[lpad2]]
// CHECK-NEXT: %[[outerpad:[^ ]*]] = cleanuppad
// CHECK: call void @"?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT: cleanupret from %[[outerpad]] unwind to caller

// CHECK-LABEL: define internal void @"?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: ret void

// CHECK-LABEL: define internal void @"?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: unreachable

void finally_within_finally() {
  __try {
    might_crash();
  } __finally {
    __try {
      might_crash();
    } __finally {
    }
  }
}

// CHECK-LABEL: define dso_local void @finally_within_finally(
// CHECK: invoke void @might_crash(

// CHECK: call void @"?fin$0@0@finally_within_finally@@"(
// CHECK: call void @"?fin$0@0@finally_within_finally@@"({{.*}}) [ "funclet"(

// CHECK-LABEL: define internal void @"?fin$0@0@finally_within_finally@@"({{[^)]*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: invoke void @might_crash(

// CHECK: call void @"?fin$1@0@finally_within_finally@@"(
// CHECK: call void @"?fin$1@0@finally_within_finally@@"({{.*}}) [ "funclet"(

// CHECK-LABEL: define internal void @"?fin$1@0@finally_within_finally@@"({{[^)]*}})
// CHECK-SAME: [[finally_attrs]]

void cleanup_with_func(const char *);
void finally_with_func() {
  __try {
    might_crash();
  } __finally {
    cleanup_with_func(__func__);
  }
}

// CHECK-LABEL: define internal void @"?fin$0@0@finally_with_func@@"({{[^)]*}})
// CHECK: call void @cleanup_with_func(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @"??_C@_0BC@COAGBPGM@finally_with_func?$AA@", i32 0, i32 0))

// Look for the absence of noinline. Enum attributes come first, so check that
// a string attribute is the first to verify that no enum attributes are
// present.
// CHECK: attributes [[finally_attrs]] = { "{{.*}}" }
