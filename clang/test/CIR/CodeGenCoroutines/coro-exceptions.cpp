// TODO(cir): drop -fno-clangir-call-conv-lowering once CallConvLowering
// supports parameters of an empty or tag class and a bare-variadic
// declaration with no named parameter.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -fno-clangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

// FIXME: we currently don't have flatten-cfg for coroutines or lower to LLVM-IR
// implemented correctly, so this tests only the CIR output.

#include "Inputs/coroutine.h"

TaskWithEH simple_eh_body() {
  co_return;
}

// CIR-LABEL: cir.func{{.*}} @_Z14simple_eh_bodyv
// CIR: }, resume : {
// Note no 'try'/'catch' here.
// CIR-NEXT: cir.call @_ZNSt15suspend_nothrow12await_resumeEv
// CIR-NEXT: cir.yield
// CIR-NEXT: }

// CIR: cir.coro.body {
// CIR-NOT: cir.if
// CIR:   cir.scope {
// CIR:     cir.try {
// CIR:       cir.call @_ZN10TaskWithEH12promise_type11return_voidEv
// CIR:       cir.co_return
// CIR:     } catch all {{.*}} {
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_ZN10TaskWithEH12promise_type19unhandled_exceptionEv
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.yield
// CIR: }
// Make sure that 'final' is outside of the above try/catch/etc.
// CIR: cir.call @_ZN10TaskWithEH12promise_type13final_suspendEv

TaskThrowingInit throwing_init_suspend() {
  co_return;
}

// CIR-LABEL: cir.func{{.*}} @_Z21throwing_init_suspendv
// CIR: %[[RESUME_FLAG:.*]] = cir.alloca "resume.eh" align(1) : !cir.ptr<!cir.bool>
// CIR: }, resume : {
// CIR:    %[[FALSE:.*]] = cir.const #false
// CIR:    cir.store %[[FALSE]], %[[RESUME_FLAG]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:    cir.scope {
// CIR:      cir.try {
// CIR:        cir.call @_ZNSt19suspend_maybe_throw12await_resumeEv
// CIR:        %[[TRUE:.*]] = cir.const #true
// CIR:        cir.store %[[TRUE]], %[[RESUME_FLAG]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:        cir.yield
// CIR:      } catch all {{.*}} {
// CIR:        cir.begin_catch
// CIR:        cir.cleanup.scope {
// CIR:          cir.call @_ZN16TaskThrowingInit12promise_type19unhandled_exceptionEv
// CIR:          cir.yield
// CIR:        } cleanup all {
// CIR:          cir.end_catch
// CIR:          cir.yield
// CIR:        }
// CIR:        cir.yield
// CIR:      }
// CIR:    }
// CIR:    cir.yield
// CIR: }

// CIR: cir.coro.body {
// CIR:   %[[LOAD_RESUME_FLAG:.*]] = cir.load align(1) %[[RESUME_FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   cir.if %[[LOAD_RESUME_FLAG]] {
// CIR:     cir.scope {
// CIR:       cir.try {
// CIR:         cir.call @_ZN16TaskThrowingInit12promise_type11return_voidEv
// CIR:         cir.co_return
// CIR:       } catch all {{.*}} {
// CIR:         cir.cleanup.scope {
// CIR:           cir.call @_ZN16TaskThrowingInit12promise_type19unhandled_exceptionEv
// CIR:           cir.yield
// CIR:         } cleanup all {
// CIR:           cir.end_catch
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.yield
// CIR:       }
// CIR:     }
// CIR:   }
// CIR:   cir.yield
// CIR: }
// CIR: cir.call @_ZN16TaskThrowingInit12promise_type13final_suspendEv
