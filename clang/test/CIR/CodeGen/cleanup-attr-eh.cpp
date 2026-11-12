// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

void cl(int *p);
void may_throw();

void f() {
  __attribute__((cleanup(cl))) int x = 1;
  may_throw();
}

// CIR: cir.func {{.*}}@_Z1fv()
// CIR:   %[[X:.*]] = cir.alloca "x"
// CIR:   cir.cleanup.scope {
// CIR:     cir.call @_Z9may_throwv()
// CIR:   } cleanup all {
// CIR:     cir.call @_Z2clPi(%[[X]]) : (!cir.ptr<!s32i> {llvm.noundef}) -> ()
// CIR:   }

// LLVM: define dso_local void @_Z1fv(){{.*}} personality ptr @__gxx_personality_v0
// LLVM: store i32 1, ptr %[[X:[^,]+]]
// LLVM: invoke void @_Z9may_throwv()
// LLVM:   to label %[[CONT:.*]] unwind label %[[LPAD:.*]]
//
// Normal path: the cleanup is a plain call.
// LLVM: [[CONT]]:
// LLVM: call void @_Z2clPi(ptr noundef %[[X]])
//
// EH path: the cleanup runs in the landing pad. A throw out of the cleanup
// itself ends up in the terminate handler.
// LLVM: [[LPAD]]:
// LLVM: landingpad { ptr, i32 }
// LLVM:   cleanup
// LLVM: invoke void @_Z2clPi(ptr noundef %[[X]])
// LLVM:   to label %[[RESUME:.*]] unwind label %[[TERMINATE:.*]]
// LLVM: resume { ptr, i32 }
// LLVM: [[TERMINATE]]:
// LLVM: landingpad { ptr, i32 }
// LLVM:   catch ptr null
// LLVM: call void @__clang_call_terminate(ptr %{{.*}})
// LLVM: unreachable

namespace ns {
void nscl(int *p);
}
template <typename T> void tcl(T *p) {}

void qualified_and_template() {
  __attribute__((cleanup(ns::nscl))) int a = 1;
  __attribute__((cleanup(tcl<int>))) int d = 1;
}

namespace ns {
void nscl(int *p) {}
}

// CIR: cir.func {{.*}}@_Z22qualified_and_templatev()
// CIR:   %[[A:.*]] = cir.alloca "a"
// CIR:   %[[D:.*]] = cir.alloca "d"
// CIR:   cir.cleanup.scope {
// CIR:     cir.cleanup.scope {
// CIR:     } cleanup all {
// CIR:       cir.call @_Z3tclIiEvPT_(%[[D]]) : (!cir.ptr<!s32i> {llvm.noundef}) -> ()
// CIR:     }
// CIR:   } cleanup all {
// CIR:     cir.call @_ZN2ns4nsclEPi(%[[A]]) : (!cir.ptr<!s32i> {llvm.noundef}) -> ()
// CIR:   }

// LLVM: define dso_local void @_Z22qualified_and_templatev(){{.*}} personality ptr @__gxx_personality_v0
// LLVM: store i32 1, ptr %[[A:[^,]+]]
// LLVM: store i32 1, ptr %[[D:[^,]+]]
//
// d is destroyed first. Its cleanup is an invoke, since a throw from it still
// has to run a's cleanup, which is the landing pad.
// LLVM: invoke void @_Z3tclIiEvPT_(ptr noundef %[[D]])
// LLVM:   to label %[[D_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[D_CONT]]:
// LLVM: call void @_ZN2ns4nsclEPi(ptr noundef %[[A]])
//
// LLVM: [[LPAD]]:
// LLVM: landingpad { ptr, i32 }
// LLVM:   cleanup
// LLVM: invoke void @_ZN2ns4nsclEPi(ptr noundef %[[A]])
// LLVM:   to label %[[RESUME:.*]] unwind label %[[TERMINATE:.*]]
// LLVM: resume { ptr, i32 }
// LLVM: [[TERMINATE]]:
// LLVM: landingpad { ptr, i32 }
// LLVM:   catch ptr null
// LLVM: call void @__clang_call_terminate(ptr %{{.*}})
