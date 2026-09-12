// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -fnamed-loops -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -fnamed-loops -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple aarch64-windows -fms-extensions -fnamed-loops -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple thumbv7-windows -fms-extensions -fnamed-loops -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck %s
// NOTE: we're passing "-O1 -disable-llvm-passes" to avoid adding optnone and noinline everywhere.

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

// CHECK-LABEL: define dso_local {{.*}}void @basic_finally()
// CHECK: invoke {{.*}}void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call {{.*}}void @"?fin$0@0@basic_finally@@"({{i8 noundef( zeroext)?}} 0, ptr noundef %[[fp]])
// CHECK-NEXT: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call {{.*}}void @"?fin$0@0@basic_finally@@"({{i8 noundef( zeroext)?}} 1, ptr noundef %[[fp]])
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK: define internal {{.*}}void @"?fin$0@0@basic_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs:#[0-9]+]]
// CHECK: call {{.*}}void @cleanup()

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

// CHECK-LABEL: define dso_local {{.*}}void @label_in_finally()
// CHECK: invoke {{.*}}void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call {{.*}}void @"?fin$0@0@label_in_finally@@"({{i8 noundef( zeroext)?}} 0, ptr noundef %[[fp]])
// CHECK: ret void

// CHECK: define internal {{.*}}void @"?fin$0@0@label_in_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br label %[[l:[^ ]*]]
//
// CHECK: [[l]]
// CHECK: call {{.*}}void @cleanup()
// CHECK: call {{.*}}i32 @check_condition()
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

// CHECK-LABEL: define dso_local {{.*}}void @use_abnormal_termination()
// CHECK: invoke {{.*}}void @might_crash()
// CHECK:     to label %[[invoke_cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[invoke_cont]]
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call {{.*}}void @"?fin$0@0@use_abnormal_termination@@"({{i8 noundef( zeroext)?}} 0, ptr noundef %[[fp]])
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call {{.*}}void @"?fin$0@0@use_abnormal_termination@@"({{i8 noundef( zeroext)?}} 1, ptr noundef %[[fp]])
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK: define internal {{.*}}void @"?fin$0@0@use_abnormal_termination@@"({{i8 noundef( zeroext)?}} %[[abnormal:abnormal_termination]], ptr noundef %frame_pointer)
// CHECK-SAME: [[finally_attrs]]
// CHECK: %[[abnormal_zext:[^ ]*]] = zext i8 %[[abnormal]] to i32
// CHECK: store i32 %[[abnormal_zext]], ptr @crashed
// CHECK-NEXT: ret void

void noreturn_noop_finally(void) {
  __try {
    __noop();
  } __finally {
    abort();
  }
}

// CHECK-LABEL: define dso_local {{.*}}void @noreturn_noop_finally()
// CHECK: call {{.*}}void @"?fin$0@0@noreturn_noop_finally@@"({{.*}})
// CHECK: ret void

// CHECK: define internal {{.*}}void @"?fin$0@0@noreturn_noop_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: call {{.*}}void @abort()
// CHECK: unreachable

void noreturn_finally(void) {
  __try {
    might_crash();
  } __finally {
    abort();
  }
}

// CHECK-LABEL: define dso_local {{.*}}void @noreturn_finally()
// CHECK: invoke {{.*}}void @might_crash()
// CHECK:     to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: call {{.*}}void @"?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: call {{.*}}void @"?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK: define internal {{.*}}void @"?fin$0@0@noreturn_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: call {{.*}}void @abort()
// CHECK: unreachable

int finally_with_return(void) {
  __try {
    return 42;
  } __finally {
  }
}
// CHECK-LABEL: define dso_local {{.*}}i32 @finally_with_return()
// CHECK: store i32 1, ptr %cleanup.dest.slot
// CHECK: %cleanup.dest = load i32, ptr %cleanup.dest.slot
// CHECK: icmp ne i32 %cleanup.dest
// CHECK: call {{.*}}void @"?fin$0@0@finally_with_return@@"({{.*}})
// CHECK: ret i32 42

// CHECK: define internal {{.*}}void @"?fin$0@0@finally_with_return@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK-NOT: br i1
// CHECK-NOT: br label
// CHECK: ret void

int nested___finally___finally(void) {
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

// CHECK-LABEL: define dso_local {{.*}}i32 @nested___finally___finally
// CHECK: invoke {{.*}}void @"?fin$1@0@nested___finally___finally@@"({{.*}})
// CHECK:          to label %[[outercont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[outercont]]
// CHECK: call {{.*}}void @"?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK-NEXT: ret i32 0
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[pad:[^ ]*]] = cleanuppad
// CHECK: call {{.*}}void @"?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@nested___finally___finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: ret void

// CHECK-LABEL: define internal {{.*}}void @"?fin$1@0@nested___finally___finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: unreachable

// FIXME: Our behavior seems suspiciously different.

int nested___finally___finally_with_eh_edge(void) {
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
// CHECK-LABEL: define dso_local {{.*}}i32 @nested___finally___finally_with_eh_edge
// CHECK: invoke {{.*}}void @might_crash()
// CHECK-NEXT: to label %[[invokecont:[^ ]*]] unwind label %[[lpad1:[^ ]*]]
//
// [[invokecont]]
// CHECK: invoke {{.*}}void @"?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT:       to label %[[outercont:[^ ]*]] unwind label %[[lpad2:[^ ]*]]
//
// CHECK: [[outercont]]
// CHECK: call {{.*}}void @"?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT: ret i32 912
//
// CHECK: [[lpad1]]
// CHECK-NEXT: %[[innerpad:[^ ]*]] = cleanuppad
// CHECK: invoke {{.*}}void @"?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT:    label %[[innercleanupretbb:[^ ]*]] unwind label %[[lpad2:[^ ]*]]
//
// CHECK: [[innercleanupretbb]]
// CHECK-NEXT: cleanupret from %[[innerpad]] unwind label %[[lpad2]]
//
// CHECK: [[lpad2]]
// CHECK-NEXT: %[[outerpad:[^ ]*]] = cleanuppad
// CHECK: call {{.*}}void @"?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-NEXT: cleanupret from %[[outerpad]] unwind to caller

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: ret void

// CHECK-LABEL: define internal {{.*}}void @"?fin$1@0@nested___finally___finally_with_eh_edge@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: unreachable

void finally_within_finally(void) {
  __try {
    might_crash();
  } __finally {
    __try {
      might_crash();
    } __finally {
    }
  }
}

// CHECK-LABEL: define dso_local {{.*}}void @finally_within_finally(
// CHECK: invoke {{.*}}void @might_crash(

// CHECK: call {{.*}}void @"?fin$0@0@finally_within_finally@@"(
// CHECK: call {{.*}}void @"?fin$0@0@finally_within_finally@@"({{.*}}) [ "funclet"(

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@finally_within_finally@@"({{[^)]*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: invoke {{.*}}void @might_crash(

// CHECK: call {{.*}}void @"?fin$1@0@finally_within_finally@@"(
// CHECK: call {{.*}}void @"?fin$1@0@finally_within_finally@@"({{.*}}) [ "funclet"(

// CHECK-LABEL: define internal {{.*}}void @"?fin$1@0@finally_within_finally@@"({{[^)]*}})
// CHECK-SAME: [[finally_attrs]]

void cleanup_with_func(const char *);
void finally_with_func(void) {
  __try {
    might_crash();
  } __finally {
    cleanup_with_func(__func__);
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@finally_with_func@@"({{[^)]*}})
// CHECK: call {{.*}}void @cleanup_with_func(ptr noundef @"??_C@_0BC@COAGBPGM@finally_with_func?$AA@")

// Jumping out of a __finally is UB, check that unreachable is emitted rather
// than a br out of __finally (which is illegal).
void goto_out_of_finally(void) {
  __try {
    might_crash();
  } __finally {
    goto out;
  }
out:
  cleanup();
}

// CHECK-LABEL: define dso_local {{.*}}void @goto_out_of_finally()
// CHECK: call {{.*}}void @"?fin$0@0@goto_out_of_finally@@"({{.*}})
// CHECK: br label %[[out:[^ ]*]]
//
// CHECK: [[out]]
// CHECK: call {{.*}}void @cleanup()
// CHECK-NEXT: ret void

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@goto_out_of_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK-NOT: br
// CHECK-NOT: ret void
// CHECK: unreachable

// Jumping inside a __finally should still be legal, so check that br is still
// emitted for jumps inside the SEH handler.
void goto_in_and_out_of_finally(void) {
before:
  __try {
    might_crash();
  } __finally {
    if (check_condition())
      goto inside;
    if (check_condition())
      goto before;
  inside:
    if (check_condition())
      goto after;
    cleanup();
  }
after:
  cleanup();
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@goto_in_and_out_of_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: call {{.*}}i32 @check_condition()
// CHECK: br {{.*}}, label %[[ifthen1:[^ ]*]], label %[[ifend1:[^ ]*]]
//
// CHECK: [[ifthen1]]
// Internal jump should still generate.
// CHECK: br label %[[inside:[^ ]*]]
//
// CHECK: [[ifend1]]
// CHECK: call {{.*}}i32 @check_condition()
// CHECK: br {{.*}}, label %[[ifthen2:[^ ]*]], label %[[ifend2:[^ ]*]]
//
// CHECK: [[ifthen2]]
// Backwards jumps still leave the helper.
// CHECK: unreachable
//
// CHECK: [[ifend2]]
// CHECK: br label %[[inside]]
//
// CHECK: [[inside]]
// CHECK: call {{.*}}i32 @check_condition()
// CHECK: br {{.*}}, label %[[ifthen3:[^ ]*]], label %[[ifend3:[^ ]*]]
//
// CHECK: [[ifthen3]]
// Forward jumps still leave the helper.
// CHECK: unreachable
//
// CHECK: [[ifend3]]
// CHECK: call {{.*}}void @cleanup()
// CHECK-NEXT: ret void

// The label can be nested arbitrarily deeply inside the __finally, but alas
// still legal and a br should still be generated.
void deep_label_in_finally(void) {
  __try {
    might_crash();
  } __finally {
    while (check_condition()) {
      switch (check_condition()) {
      case 1:
      deep:
        cleanup();
        break;
      default:
        goto deep;
      }
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@deep_label_in_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: switch i32 %{{[^,]*}}, label %[[swdefault:[^ ]*]] [
// CHECK: br label %[[deep:[^ ]*]]
//
// CHECK: [[deep]]
// CHECK: call {{.*}}void @cleanup()
//
// CHECK: [[swdefault]]
// CHECK-NEXT: br label %[[deep]]
// CHECK-NOT: unreachable
// CHECK: ret void

void break_out_of_finally(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      break;
    }
  }
}

// CHECK-LABEL: define dso_local {{.*}}void @break_out_of_finally()
// CHECK: call {{.*}}void @"?fin$0@0@break_out_of_finally@@"({{.*}})

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@break_out_of_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// break out of an outlined SEH handler is UB, do not generate br.
// CHECK-NOT: br
// CHECK-NOT: ret void
// CHECK: unreachable

void break_out_of_finally_to_switch(void) {
  switch (check_condition()) {
  case 1:
    __try {
      might_crash();
    } __finally {
      break;
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@break_out_of_finally_to_switch@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK-NOT: br
// CHECK-NOT: ret void
// CHECK: unreachable

void continue_out_of_finally(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      continue;
    }
  }
}

// CHECK-LABEL: define dso_local {{.*}}void @continue_out_of_finally()
// CHECK: call {{.*}}void @"?fin$0@0@continue_out_of_finally@@"({{.*}})

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@continue_out_of_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// continue out of an outlined SEH handler is UB, do not generate br.
// CHECK-NOT: br
// CHECK-NOT: ret void
// CHECK: unreachable

// break/continue bound to a loop within an outlined SEH handler should still
// generate br's as normal.
void loop_inside_finally(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      while (check_condition()) {
        if (check_condition())
          continue;
        break;
      }
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@loop_inside_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br label %[[whilecond:[^ ]*]]
//
// CHECK: [[whilecond]]
// CHECK: br i1 {{.*}}, label %{{[^ ]*}}, label %[[whileend:[^ ]*]]
//
// CHECK: br label %[[whilecond]]
// CHECK: br label %[[whileend]]
//
// CHECK: [[whileend]]
// CHECK-NOT: unreachable
// CHECK: ret void

// Ditto.
void break_in_switch_in_finally(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      switch (check_condition()) {
      case 1:
        break;
      }
      cleanup();
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@break_in_switch_in_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: switch i32 %{{[^,]*}}, label %[[epilog:[^ ]*]] [
// CHECK-NEXT: i32 1, label %[[epilog]]
//
// CHECK: [[epilog]]
// CHECK-NOT: unreachable
// CHECK: call {{.*}}void @cleanup()
// CHECK-NEXT: ret void

void switch_in_loop_in_finally(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      while (check_condition()) {
        switch (check_condition()) {
        case 1:
          continue;
        }
      }
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@switch_in_loop_in_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br label %[[whilecond:[^ ]*]]
// CHECK: switch i32 %{{[^,]*}}, label %{{[^ ]*}} [
// CHECK-NEXT: i32 1, label %[[swbb:[^ ]*]]
//
// CHECK: [[swbb]]
// CHECK-NEXT: br label %[[whilecond]]
// CHECK-NOT: unreachable
// CHECK: ret void

void continue_in_switch_in_finally(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      switch (check_condition()) {
      case 1:
        continue;
      }
      cleanup();
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@continue_in_switch_in_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: switch i32 %{{[^,]*}}, label %[[epilog:[^ ]*]] [
// CHECK-NEXT: i32 1, label %[[swbb:[^ ]*]]
//
// CHECK: [[swbb]]
// continue corresponds to the loop instead, so generate unreachable.
// CHECK-NEXT: unreachable
//
// CHECK: [[epilog]]
// CHECK: call {{.*}}void @cleanup()
// CHECK-NEXT: ret void

// C2Y named loops: This is still UB. 
void named_break_out_of_finally(void) {
outer:
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      while (check_condition())
        break outer;
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@named_break_out_of_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br label %[[whilecond:[^ ]*]]
//
// CHECK: [[whilecond]]
// CHECK: br i1 {{.*}}, label %[[whilebody:[^ ]*]], label %[[whileend:[^ ]*]]
//
// CHECK: [[whilebody]]
// CHECK-NEXT: unreachable
//
// CHECK: [[whileend]]
// CHECK-NEXT: ret void

// C2Y named loops: This is still UB.
void named_continue_out_of_finally(void) {
outer:
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      while (check_condition())
        continue outer;
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@named_continue_out_of_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br label %[[whilecond:[^ ]*]]
//
// CHECK: [[whilecond]]
// CHECK: br i1 {{.*}}, label %[[whilebody:[^ ]*]], label %[[whileend:[^ ]*]]
//
// CHECK: [[whilebody]]
// CHECK-NEXT: unreachable
//
// CHECK: [[whileend]]
// CHECK-NEXT: ret void

// C2Y named loops: Named loops within an outlined SEH handler should not
// generate unreachable's.
void named_break_inside_finally(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
    inner:
      while (check_condition())
        while (check_condition())
          break inner;
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@named_break_inside_finally@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br i1 {{.*}}, label %{{[^ ]*}}, label %[[outerend:[^ ]*]]
// CHECK: br i1 {{.*}}, label %[[innerbody:[^ ]*]], label %{{[^ ]*}}
//
// CHECK: [[innerbody]]
// CHECK-NEXT: br label %[[outerend]]
// CHECK-NOT: unreachable
// CHECK: [[outerend]]
// CHECK: ret void

// The break's loop dest is in the outer __finally, generate unreachable in the
// inner loop while leaving outer loop intact.
void nested_finally_break(void) {
  for (int i = 0; i < 4; i++) {
    __try {
      might_crash();
    } __finally {
      while (check_condition()) {
        __try {
          might_crash();
        } __finally {
          break;
        }
      }
    }
  }
}

// CHECK-LABEL: define internal {{.*}}void @"?fin$0@0@nested_finally_break@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK: br label %[[whilecond:[^ ]*]]
// CHECK: call {{.*}}void @"?fin$1@0@nested_finally_break@@"({{.*}})
// CHECK: br label %[[whilecond]]
// CHECK-NOT: unreachable
// CHECK: ret void

// CHECK-LABEL: define internal {{.*}}void @"?fin$1@0@nested_finally_break@@"({{.*}})
// CHECK-SAME: [[finally_attrs]]
// CHECK-NOT: br
// CHECK-NOT: ret void
// CHECK: unreachable

// Look for the absence of noinline.  nounwind is expected; any further
// attributes should be string attributes.
// CHECK: attributes [[finally_attrs]] = { nounwind "{{.*}}" }
