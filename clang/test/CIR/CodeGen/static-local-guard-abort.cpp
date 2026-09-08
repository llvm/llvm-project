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

struct Temp {
  Temp(int);
  ~Temp();
};

struct Elem {
  Elem(const Temp &);
  ~Elem();
};

void g() { static Elem arr[] = {Temp(1), Temp(2), Temp(3)}; }

// An array element constructor can throw part-way through, which stacks three
// kinds of cleanup inside the guard. Outermost is the guard-abort scope; within
// it the partial-array destroy, EH-only and bounded by the arrayinit.endOfInit
// watermark so it destroys exactly the elements already constructed; and within
// that a `cleanup all` per argument temporary, which is destroyed on the normal
// path too. The __cxa_atexit that registers the whole array stays inside the
// guard-abort scope, and only __cxa_guard_release is outside it.

// CIR-LABEL: cir.func {{.*}}@_Z1gv
// CIR:         %[[G:.*]] = cir.get_global @_ZGVZ1gvE3arr
// CIR:         cir.call @__cxa_guard_acquire(%[[G]])
// CIR:         cir.cleanup.scope {
// CIR:           %[[END:.*]] = cir.alloca "arrayinit.endOfInit"
// CIR:           cir.cleanup.scope {
// CIR:             cir.call @_ZN4TempC1Ei
// CIR:             cir.call @_ZN4ElemC1ERK4Temp
// CIR:           } cleanup all {
// CIR:             cir.call @_ZN4TempD1Ev({{.*}}) nothrow
// CIR:           } cleanup eh {
// CIR:             cir.load{{.*}} %[[END]]
// CIR:             cir.call @_ZN4ElemD1Ev({{.*}}) nothrow
// CIR:           cir.call @__cxa_atexit
// CIR:         } cleanup eh {
// CIR:           cir.call @__cxa_guard_abort(%[[G]]) nothrow
// CIR:         cir.call @__cxa_guard_release(%[[G]])

// LLVM-LABEL: define {{.*}}void @_Z1gv()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ1gvE3arr)
// LLVM:         invoke void @_ZN4ElemC1ERK4Temp
// LLVM:         call void @_ZN4ElemD1Ev
// LLVM:         call void @__cxa_guard_abort(ptr @_ZGVZ1gvE3arr)
// LLVM:         resume { ptr, i32 }

// NOEXC-LABEL: define {{.*}}void @_Z1gv()
// NOEXC-NOT:     __cxa_guard_abort
// NOEXC-NOT:     landingpad
// NOEXC:         ret void
