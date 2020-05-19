// REQUIRES: webassembly-registered-target
// https://reviews.llvm.org/D79655 temporarily added a RUN line that was missing
// a -o flag and wrote to the source dir. The file it wrote was then interpreted
// as a test without RUN line, breaking bots. FIXME: Remove this rm line once
// it's been in the tree long enough to clean up everyone's build dirs.
// Removing this June 2020 should be fine.
// RUN: rm -f %S/wasm-eh.ll
// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -fms-extensions -fexceptions -fcxx-exceptions -fwasm-exceptions -target-feature +exception-handling -emit-llvm -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 %s -triple wasm64-unknown-unknown -fms-extensions -fexceptions -fcxx-exceptions -fwasm-exceptions -target-feature +exception-handling -emit-llvm -o - -std=c++11 | FileCheck %s

void may_throw();
void dont_throw() noexcept;

struct Cleanup {
  ~Cleanup() { dont_throw(); }
};

// Multiple catch clauses w/o catch-all
void test0() {
  try {
    may_throw();
  } catch (int) {
    dont_throw();
  } catch (double) {
    dont_throw();
  }
}

// CHECK-LABEL: define void @_Z5test0v() {{.*}} personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*)

// CHECK:   %[[INT_ALLOCA:.*]] = alloca i32
// CHECK:   invoke void @_Z9may_throwv()
// CHECK-NEXT:           to label %[[NORMAL_BB:.*]] unwind label %[[CATCHDISPATCH_BB:.*]]

// CHECK: [[CATCHDISPATCH_BB]]:
// CHECK-NEXT:   %[[CATCHSWITCH:.*]] = catchswitch within none [label %[[CATCHSTART_BB:.*]]] unwind to caller

// CHECK: [[CATCHSTART_BB]]:
// CHECK-NEXT:   %[[CATCHPAD:.*]] = catchpad within %[[CATCHSWITCH]] [i8* bitcast (i8** @_ZTIi to i8*), i8* bitcast (i8** @_ZTId to i8*)]
// CHECK-NEXT:   %[[EXN:.*]] = call i8* @llvm.wasm.get.exception(token %[[CATCHPAD]])
// CHECK-NEXT:   store i8* %[[EXN]], i8** %exn.slot
// CHECK-NEXT:   %[[SELECTOR:.*]] = call i32 @llvm.wasm.get.ehselector(token %[[CATCHPAD]])
// CHECK-NEXT:   %[[TYPEID:.*]] = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2
// CHECK-NEXT:   %[[MATCHES:.*]] = icmp eq i32 %[[SELECTOR]], %[[TYPEID]]
// CHECK-NEXT:   br i1 %[[MATCHES]], label %[[CATCH_INT_BB:.*]], label %[[CATCH_FALLTHROUGH_BB:.*]]

// CHECK: [[CATCH_INT_BB]]:
// CHECK-NEXT:   %[[EXN:.*]] = load i8*, i8** %exn.slot
// CHECK-NEXT:   %[[ADDR:.*]] = call i8* @__cxa_begin_catch(i8* %[[EXN]]) {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:   %[[ADDR_CAST:.*]] = bitcast i8* %[[ADDR]] to i32*
// CHECK-NEXT:   %[[INT_VAL:.*]] = load i32, i32* %[[ADDR_CAST]]
// CHECK-NEXT:   store i32 %[[INT_VAL]], i32* %[[INT_ALLOCA]]
// CHECK-NEXT:   call void @_Z10dont_throwv() {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:   call void @__cxa_end_catch() {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:   catchret from %[[CATCHPAD]] to label %[[CATCHRET_DEST_BB0:.*]]

// CHECK: [[CATCHRET_DEST_BB0]]:
// CHECK-NEXT:   br label %[[TRY_CONT_BB:.*]]

// CHECK: [[CATCH_FALLTHROUGH_BB]]
// CHECK-NEXT:   %[[TYPEID:.*]] = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTId to i8*)) #2
// CHECK-NEXT:   %[[MATCHES:.*]] = icmp eq i32 %[[SELECTOR]], %[[TYPEID]]
// CHECK-NEXT:   br i1 %[[MATCHES]], label %[[CATCH_FLOAT_BB:.*]], label %[[RETHROW_BB:.*]]

// CHECK: [[CATCH_FLOAT_BB]]:
// CHECK:   catchret from %[[CATCHPAD]] to label %[[CATCHRET_DEST_BB1:.*]]

// CHECK: [[CATCHRET_DEST_BB1]]:
// CHECK-NEXT:   br label %[[TRY_CONT_BB]]

// CHECK: [[RETHROW_BB]]:
// CHECK-NEXT:   call void @llvm.wasm.rethrow.in.catch() {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:   unreachable

// Single catch-all
void test1() {
  try {
    may_throw();
  } catch (...) {
    dont_throw();
  }
}

// CATCH-LABEL: @_Z5test1v()

// CHECK:   %[[CATCHSWITCH:.*]] = catchswitch within none [label %[[CATCHSTART_BB:.*]]] unwind to caller

// CHECK: [[CATCHSTART_BB]]:
// CHECK-NEXT:   %[[CATCHPAD:.*]] = catchpad within %[[CATCHSWITCH]] [i8* null]
// CHECK:   br label %[[CATCH_ALL_BB:.*]]

// CHECK: [[CATCH_ALL_BB]]:
// CHECK:   catchret from %[[CATCHPAD]] to label

// Multiple catch clauses w/ catch-all
void test2() {
  try {
    may_throw();
  } catch (int) {
    dont_throw();
  } catch (...) {
    dont_throw();
  }
}

// CHECK-LABEL: @_Z5test2v()

// CHECK:   %[[CATCHSWITCH:.*]] = catchswitch within none [label %[[CATCHSTART_BB:.*]]] unwind to caller

// CHECK: [[CATCHSTART_BB]]:
// CHECK-NEXT:   %[[CATCHPAD:.*]] = catchpad within %[[CATCHSWITCH]] [i8* bitcast (i8** @_ZTIi to i8*), i8* null]
// CHECK:   br i1 %{{.*}}, label %[[CATCH_INT_BB:.*]], label %[[CATCH_ALL_BB:.*]]

// CHECK: [[CATCH_INT_BB]]:
// CHECK:   catchret from %[[CATCHPAD]] to label

// CHECK: [[CATCH_ALL_BB]]:
// CHECK:   catchret from %[[CATCHPAD]] to label

// Cleanup
void test3() {
  Cleanup c;
  may_throw();
}

// CHECK-LABEL: @_Z5test3v()

// CHECK:   invoke void @_Z9may_throwv()
// CHECK-NEXT:           to label {{.*}} unwind label %[[EHCLEANUP_BB:.*]]

// CHECK: [[EHCLEANUP_BB]]:
// CHECK-NEXT:   %[[CLEANUPPAD:.*]] = cleanuppad within none []
// CHECK-NEXT:   call %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %{{.*}}) {{.*}} [ "funclet"(token %[[CLEANUPPAD]]) ]
// CHECK-NEXT:   cleanupret from %[[CLEANUPPAD]] unwind to caller

// Possibly throwing function call within a catch
void test4() {
  try {
    may_throw();
  } catch (int) {
    may_throw();
  }
}

// CHECK-LABEL: @_Z5test4v()

// CHECK:   %[[CATCHSWITCH]] = catchswitch within none [label %[[CATCHSTART_BB]]] unwind to caller

// CHECK: [[CATCHSTART_BB]]:
// CHECK:   %[[CATCHPAD:.*]] = catchpad within %[[CATCHSWITCH]] [i8* bitcast (i8** @_ZTIi to i8*)]

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:           to label %[[INVOKE_CONT_BB:.*]] unwind label %[[EHCLEANUP_BB:.*]]

// CHECK: [[INVOKE_CONT_BB]]:
// CHECK-NEXT:   call void @__cxa_end_catch() {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:   catchret from %[[CATCHPAD]] to label

// CHECK: [[EHCLEANUP_BB]]:
// CHECK-NEXT:   %[[CLEANUPPAD:.*]] = cleanuppad within %[[CATCHPAD]] []
// CHECK-NEXT:   call void @__cxa_end_catch() {{.*}} [ "funclet"(token %[[CLEANUPPAD]]) ]
// CHECK-NEXT:   cleanupret from %[[CLEANUPPAD]] unwind to caller

// Possibly throwing function call within a catch-all
void test5() {
  try {
    may_throw();
  } catch (...) {
    may_throw();
  }
}

// CHECK-LABEL: @_Z5test5v()

// CHECK:   %[[CATCHSWITCH:.*]] = catchswitch within none [label %[[CATCHSTART_BB]]] unwind to caller

// CHECK: [[CATCHSTART_BB]]:
// CHECK:   %[[CATCHPAD:.*]] = catchpad within %[[CATCHSWITCH]] [i8* null]

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:           to label %[[INVOKE_CONT_BB0:.*]] unwind label %[[EHCLEANUP_BB:.*]]

// CHECK: [[INVOKE_CONT_BB0]]:
// CHECK-NEXT:   call void @__cxa_end_catch() [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:   catchret from %[[CATCHPAD]] to label

// CHECK: [[EHCLEANUP_BB]]:
// CHECK-NEXT:   %[[CLEANUPPAD0:.*]] = cleanuppad within %[[CATCHPAD]] []
// CHECK-NEXT:   invoke void @__cxa_end_catch() [ "funclet"(token %[[CLEANUPPAD0]]) ]
// CHECK-NEXT:           to label %[[INVOKE_CONT_BB1:.*]] unwind label %[[TERMINATE_BB:.*]]

// CHECK: [[INVOKE_CONT_BB1]]:
// CHECK-NEXT:   cleanupret from %[[CLEANUPPAD0]] unwind to caller

// CHECK: [[TERMINATE_BB]]:
// CHECK-NEXT:   %[[CLEANUPPAD1:.*]] = cleanuppad within %[[CLEANUPPAD0]] []
// CHECK-NEXT:   %[[EXN:.*]] = call i8* @llvm.wasm.get.exception(token %[[CLEANUPPAD1]])
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %[[EXN]]) {{.*}} [ "funclet"(token %[[CLEANUPPAD1]]) ]
// CHECK-NEXT:   unreachable

// CHECK-LABEL: define {{.*}} void @__clang_call_terminate(i8* %0)
// CHECK-NEXT:   call i8* @__cxa_begin_catch(i8* %{{.*}})
// CHECK-NEXT:   call void @_ZSt9terminatev()
// CHECK-NEXT:   unreachable

// Try-catch with cleanups
void test6() {
  Cleanup c1;
  try {
    Cleanup c2;
    may_throw();
  } catch (int) {
    Cleanup c3;
    may_throw();
  }
}

// CHECK-LABEL: @_Z5test6v()
// CHECK:   invoke void @_Z9may_throwv()
// CHECK-NEXT:           to label %{{.*}} unwind label %[[EHCLEANUP_BB0:.*]]

// CHECK: [[EHCLEANUP_BB0]]:
// CHECK-NEXT:   %[[CLEANUPPAD0:.*]] = cleanuppad within none []
// CHECK-NEXT:   call %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* {{.*}}) {{.*}} [ "funclet"(token %[[CLEANUPPAD0]]) ]
// CHECK-NEXT:   cleanupret from %[[CLEANUPPAD0]] unwind label %[[CATCH_DISPATCH_BB:.*]]

// CHECK: [[CATCH_DISPATCH_BB]]:
// CHECK-NEXT:  %[[CATCHSWITCH:.*]] = catchswitch within none [label %[[CATCHSTART_BB:.*]]] unwind label %[[EHCLEANUP_BB1:.*]]

// CHECK: [[CATCHSTART_BB]]:
// CHECK-NEXT:   %[[CATCHPAD:.*]] = catchpad within %[[CATCHSWITCH]] [i8* bitcast (i8** @_ZTIi to i8*)]
// CHECK:   br i1 %{{.*}}, label %[[CATCH_INT_BB:.*]], label %[[RETHROW_BB:.*]]

// CHECK: [[CATCH_INT_BB]]:
// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:           to label %[[INVOKE_CONT_BB:.*]] unwind label %[[EHCLEANUP_BB2:.*]]

// CHECK: [[INVOKE_CONT_BB]]:
// CHECK:   catchret from %[[CATCHPAD]] to label %{{.*}}

// CHECK: [[RETHROW_BB]]:
// CHECK-NEXT:   invoke void @llvm.wasm.rethrow.in.catch() {{.*}} [ "funclet"(token %[[CATCHPAD]]) ]
// CHECK-NEXT:          to label %[[UNREACHABLE_BB:.*]] unwind label %[[EHCLEANUP_BB1:.*]]

// CHECK: [[EHCLEANUP_BB2]]:
// CHECK-NEXT:   %[[CLEANUPPAD2:.*]] = cleanuppad within %[[CATCHPAD]] []
// CHECK-NEXT:   call %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %{{.*}}) {{.*}} [ "funclet"(token %[[CLEANUPPAD2]]) ]
// CHECK-NEXT:   cleanupret from %[[CLEANUPPAD2]] unwind label %[[EHCLEANUP_BB3:.*]]

// CHECK: [[EHCLEANUP_BB3]]:
// CHECK-NEXT:   %[[CLEANUPPAD3:.*]] = cleanuppad within %[[CATCHPAD]] []
// CHECK:   cleanupret from %[[CLEANUPPAD3]] unwind label %[[EHCLEANUP_BB1:.*]]

// CHECK: [[EHCLEANUP_BB1]]:
// CHECK-NEXT:   %[[CLEANUPPAD1:.*]] = cleanuppad within none []
// CHECK-NEXT:   call %struct.Cleanup* @_ZN7CleanupD1Ev(%struct.Cleanup* %{{.*}}) {{.*}} [ "funclet"(token %[[CLEANUPPAD1]]) ]
// CHECK-NEXT:   cleanupret from %[[CLEANUPPAD1]] unwind to caller

// CHECK: [[UNREACHABLE_BB]]:
// CHECK-NEXT:   unreachable

// Nested try-catches within a try with cleanups
void test7() {
  Cleanup c1;
  may_throw();
  try {
    Cleanup c2;
    may_throw();
    try {
      Cleanup c3;
      may_throw();
    } catch (int) {
      may_throw();
    } catch (double) {
      may_throw();
    }
  } catch (int) {
    may_throw();
  } catch (...) {
    may_throw();
  }
}

// CHECK-LABEL: @_Z5test7v()
// CHECK:   invoke void @_Z9may_throwv()

// CHECK:   invoke void @_Z9may_throwv()

// CHECK:   invoke void @_Z9may_throwv()

// CHECK:   %[[CLEANUPPAD0:.*]] = cleanuppad within none []
// CHECK:   cleanupret from %[[CLEANUPPAD0]] unwind label

// CHECK:   %[[CATCHSWITCH0:.*]] = catchswitch within none

// CHECK:   %[[CATCHPAD0:.*]] = catchpad within %[[CATCHSWITCH0]] [i8* bitcast (i8** @_ZTIi to i8*), i8* bitcast (i8** @_ZTId to i8*)]

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD0]]) ]

// CHECK:   catchret from %[[CATCHPAD0]] to label

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD0]]) ]

// CHECK:   catchret from %[[CATCHPAD0]] to label

// CHECK:   invoke void @llvm.wasm.rethrow.in.catch() {{.*}} [ "funclet"(token %[[CATCHPAD0]]) ]

// CHECK:   %[[CLEANUPPAD1:.*]] = cleanuppad within %[[CATCHPAD0]] []
// CHECK:   cleanupret from %[[CLEANUPPAD1]] unwind label

// CHECK:   %[[CLEANUPPAD2:.*]] = cleanuppad within %[[CATCHPAD0]] []
// CHECK:   cleanupret from %[[CLEANUPPAD2]] unwind label

// CHECK:   %[[CLEANUPPAD3:.*]] = cleanuppad within none []
// CHECK:   cleanupret from %[[CLEANUPPAD3]] unwind label

// CHECK:   %[[CATCHSWITCH1:.*]] = catchswitch within none

// CHECK:   %[[CATCHPAD1:.*]] = catchpad within %[[CATCHSWITCH1]] [i8* bitcast (i8** @_ZTIi to i8*), i8* null]

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD1]]) ]

// CHECK:   catchret from %[[CATCHPAD1]] to label

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD1]]) ]

// CHECK:   invoke void @__cxa_end_catch() [ "funclet"(token %[[CATCHPAD1]]) ]

// CHECK:   catchret from %[[CATCHPAD1]] to label

// CHECK:   %[[CLEANUPPAD4:.*]] = cleanuppad within %[[CATCHPAD1]] []
// CHECK:   invoke void @__cxa_end_catch() [ "funclet"(token %[[CLEANUPPAD4]]) ]

// CHECK:   cleanupret from %[[CLEANUPPAD4]] unwind label

// CHECK:   %[[CLEANUPPAD5:.*]] = cleanuppad within %[[CATCHPAD1]] []
// CHECK:   cleanupret from %[[CLEANUPPAD5]] unwind label

// CHECK:   %[[CLEANUPPAD6:.*]] = cleanuppad within none []
// CHECK:   cleanupret from %[[CLEANUPPAD6]] unwind to caller

// CHECK:   unreachable

// CHECK:   %[[CLEANUPPAD7:.*]] = cleanuppad within %[[CLEANUPPAD4]] []
// CHECK:   call void @__clang_call_terminate(i8* %{{.*}}) {{.*}} [ "funclet"(token %[[CLEANUPPAD7]]) ]
// CHECK:   unreachable

// Nested try-catches within a catch
void test8() {
  try {
    may_throw();
  } catch (int) {
    try {
      may_throw();
    } catch (int) {
      may_throw();
    }
  }
}

// CHECK-LABEL: @_Z5test8v()
// CHECK:   invoke void @_Z9may_throwv()

// CHECK:   %[[CATCHSWITCH0:.*]] = catchswitch within none

// CHECK:   %[[CATCHPAD0:.*]] = catchpad within %[[CATCHSWITCH0]] [i8* bitcast (i8** @_ZTIi to i8*)]

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD0]]) ]

// CHECK:   %[[CATCHSWITCH1:.*]] = catchswitch within %[[CATCHPAD0]]

// CHECK:   %[[CATCHPAD1:.*]] = catchpad within %[[CATCHSWITCH1]] [i8* bitcast (i8** @_ZTIi to i8*)]

// CHECK:   invoke void @_Z9may_throwv() [ "funclet"(token %[[CATCHPAD1]]) ]

// CHECK:   catchret from %[[CATCHPAD1]] to label

// CHECK:   invoke void @llvm.wasm.rethrow.in.catch() {{.*}} [ "funclet"(token %[[CATCHPAD1]]) ]

// CHECK:   catchret from %[[CATCHPAD0]] to label

// CHECK:   call void @llvm.wasm.rethrow.in.catch() {{.*}} [ "funclet"(token %[[CATCHPAD0]]) ]
// CHECK:   unreachable

// CHECK:   %[[CLEANUPPAD0:.*]] = cleanuppad within %[[CATCHPAD1]] []
// CHECK:   cleanupret from %[[CLEANUPPAD0]] unwind label

// CHECK:   %[[CLEANUPPAD1:.*]] = cleanuppad within %[[CATCHPAD0]] []
// CHECK:   cleanupret from %[[CLEANUPPAD1]] unwind to caller

// CHECK:   unreachable

// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -fms-extensions -fexceptions -fcxx-exceptions -fwasm-exceptions -target-feature +exception-handling -emit-llvm -o - -std=c++11 2>&1 | FileCheck %s --check-prefix=WARNING-DEFAULT
// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -fms-extensions -fexceptions -fcxx-exceptions -fwasm-exceptions -target-feature +exception-handling -Wwasm-exception-spec -emit-llvm -o - -std=c++11 2>&1 | FileCheck %s --check-prefix=WARNING-ON
// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -fms-extensions -fexceptions -fcxx-exceptions -fwasm-exceptions -target-feature +exception-handling -Wno-wasm-exception-spec -emit-llvm -o - -std=c++11 2>&1 | FileCheck %s --check-prefix=WARNING-OFF

// Wasm ignores dynamic exception specifications with types at the moment. This
// is controlled by -Wwasm-exception-spec, which is on by default. This warning
// can be suppressed with -Wno-wasm-exception-spec.
// Checks if a warning message is correctly printed or not printed depending on
// the options.
void test9() throw(int) {
}
// WARNING-DEFAULT: warning: dynamic exception specifications with types are currently ignored in wasm
// WARNING-ON: warning: dynamic exception specifications with types are currently ignored in wasm
// WARNING-OFF-NOT: warning: dynamic exception specifications with types are currently ignored in wasm

// Wasm curremtly treats 'throw()' in the same way as 'noexept'. Check if the
// same warning message is printed as if when a 'noexcept' function throws.
void test10() throw() {
  throw 3;
}
// WARNING-DEFAULT: warning: 'test10' has a non-throwing exception specification but can still throw
// WARNING-DEFAULT: function declared non-throwing here

// Here we only check if the command enables wasm exception handling in the
// backend so that exception handling instructions can be generated in .s file.

// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -fms-extensions -fexceptions -fcxx-exceptions -fwasm-exceptions -target-feature +exception-handling -S -o - -std=c++11 | FileCheck %s --check-prefix=ASSEMBLY

// ASSEMBLY: try
// ASSEMBLY: catch
// ASSEMBLY: rethrow
// ASSEMBLY: end_try
