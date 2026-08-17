// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=NOEXC
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fno-threadsafe-statics -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=NOTSS

// An exception out of a block-scope static's initializer leaves it
// uninitialized and the next entry has to run the initializer again
// ([stmt.dcl]p4), which the Itanium ABI implements by handing the guard back
// with __cxa_guard_abort on the unwind path. Without it the guard stays held
// and that retry terminates.

struct S {
  S();
  ~S();
};

void f() { static S s; }

// The initializer -- and the __cxa_atexit registration that follows it -- sits
// in the body of an EH-only cleanup scope holding the abort call.

// CIR-LABEL: cir.func {{.*}}@_Z1fv
// CIR:         %[[GUARD:.*]] = cir.get_global @_ZGVZ1fvE1s
// CIR:         cir.call @__cxa_guard_acquire(%[[GUARD]])
// CIR:         cir.cleanup.scope {
// CIR:           cir.call @_ZN1SC1Ev
// CIR:           cir.call @__cxa_atexit
// CIR:         } cleanup eh {
// CIR:           cir.call @__cxa_guard_abort(%[[GUARD]]) nothrow
// CIR:         }
// CIR:         cir.call @__cxa_guard_release(%[[GUARD]])

// LLVM-LABEL: define {{.*}}void @_Z1fv()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ1fvE1s)
// LLVM:         invoke void @_ZN1SC1Ev
// LLVM:         landingpad { ptr, i32 }
// LLVM:           cleanup
// LLVM:         call void @__cxa_guard_abort(ptr @_ZGVZ1fvE1s)
// LLVM:         resume { ptr, i32 }

// With exceptions off there is no unwind path to hand the guard back on, and
// building the cleanup scope anyway would manufacture one.

// NOEXC-LABEL: define {{.*}}void @_Z1fv()
// NOEXC-NOT:     __cxa_guard_abort
// NOEXC-NOT:     landingpad
// NOEXC:         ret void

// Without thread-safe statics there is no lock to release: the guard byte is
// simply stored after the initializer completes, so an exception skips it.

// NOTSS-LABEL: define {{.*}}void @_Z1fv()
// NOTSS-NOT:     __cxa_guard_abort
// NOTSS-NOT:     __cxa_guard_acquire
// NOTSS:         ret void
