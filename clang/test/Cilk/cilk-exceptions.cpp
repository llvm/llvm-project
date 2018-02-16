// Test case for code generation of Tapir for Cilk code that uses exceptions.
//
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fcilkplus -ftapir=none -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm %s -o - | FileCheck %s

void handle_exn(int e = -1);

class Foo {
public:
  Foo() {}
  ~Foo() {}
};

int bar(Foo *f);
int quuz(int i) noexcept;
__attribute__((always_inline))
int foo(Foo *f) {
  try
    {
      bar(f);
    }
  catch (int e)
    {
      handle_exn(e);
    }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Serial code snippets
////////////////////////////////////////////////////////////////////////////////

// CHECK-LABEL: @_Z15serial_noexcepti(
// CHECK-NOT: sync
void serial_noexcept(int n) {
  quuz(n);
  quuz(n);
}

// CHECK-LABEL: @_Z13serial_excepti(
// CHECK-NOT: sync
void serial_except(int n) {
  bar(new Foo());
  quuz(n);
}

// CHECK-LABEL: @_Z15serial_tryblocki(
// CHECK-NOT: sync
void serial_tryblock(int n) {
  try
    {
      quuz(n);
      bar(new Foo());
      quuz(n);
      bar(new Foo());
    }
  catch (int e)
    {
      handle_exn(e);
    }
  catch (...)
    {
      handle_exn();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// _Cilk_for code snippets
////////////////////////////////////////////////////////////////////////////////

// CHECK-LABEL: @_Z20parallelfor_noexcepti(
// CHECK-NOT: detach within %{{.+}}, label %{{.+}}, label %{{.+}} unwind
// CHECK-NOT: landingpad
// CHECK-NOT: resume
void parallelfor_noexcept(int n) {
  _Cilk_for (int i = 0; i < n; ++i)
    quuz(i);
}

// CHECK-LABEL: @_Z18parallelfor_excepti(
// CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
// CHECK-DAG: sync within %[[SYNCREG]]
// CHECK: detach within %[[SYNCREG]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind label %[[DUNWIND:.+]]
// CHECK: invoke i8* @_Znwm(i64 1)
// CHECK-NEXT: to label %[[INVOKECONT1:.+]] unwind label %[[TASKLPAD1:.+]]
// CHECK: [[INVOKECONT1]]:
// CHECK: invoke void @_ZN3FooC1Ev(
// CHECK-NEXT: to label %[[INVOKECONT2:.+]] unwind label %[[TASKLPAD2:.+]]
// CHECK: [[INVOKECONT2]]:
// CHECK: invoke i32 @_Z3barP3Foo(
// CHECK-NEXT: to label %[[INVOKECONT3:.+]] unwind label %[[TASKLPAD1]]
// CHECK: [[TASKLPAD1]]:
// CHECK-NEXT: landingpad [[LPADTYPE:.+]]
// CHECK-NEXT: catch {{.+}} null
// CHECK: [[TASKLPAD2]]:
// CHECK-NEXT: landingpad [[LPADTYPE]]
// CHECK-NEXT: catch {{.+}} null
// CHECK: invoke void @llvm.detached.rethrow
// CHECK: (token %[[SYNCREG]], [[LPADTYPE]] {{.+}})
// CHECK-NEXT: to label %[[DRUNREACH:.+]] unwind label %[[DUNWIND]]
// CHECK: [[DRUNREACH]]:
// CHECK-NEXT: unreachable
// CHECK: [[DUNWIND]]:
// CHECK-NEXT: landingpad [[LPADTYPE]]
// CHECK: sync within %[[SYNCREG]]
void parallelfor_except(int n) {
  _Cilk_for (int i = 0; i < n; ++i)
    bar(new Foo());
}

// CHECK-LABEL: @_Z20parallelfor_tryblocki(
void parallelfor_tryblock(int n) {
  // CHECK: %[[SYNCREG1:.+]] = call token @llvm.syncregion.start()
  // CHECK: %[[SYNCREG2:.+]] = call token @llvm.syncregion.start()
  try
    {
      // CHECK-NOT: detach within %[[SYNCREG1]], label %{{.+}}, label %{{.+}} unwind
      _Cilk_for (int i = 0; i < n; ++i)
        quuz(i);

      // CHECK: detach within %[[SYNCREG2]], label %[[DETACHED:.+]], label %{{.+}} unwind label %[[DUNWIND:.+]]
      // CHECK: [[DETACHED]]:
      // CHECK: invoke
      // CHECK-NEXT: to label %[[INVOKECONT1:.+]] unwind label %[[TASKLPAD:.+]]
      // CHECK: [[TASKLPAD]]:
      // CHECK-NEXT: landingpad [[LPADTYPE:.+]]
      // CHECK-NEXT: catch {{.+}} null
      // CHECK: invoke void @llvm.detached.rethrow
      // CHECK: (token %[[SYNCREG2]], [[LPADTYPE]] {{.+}})
      // CHECK-NEXT: to label {{.+}} unwind label %[[DUNWIND]]
      // CHECK: [[DUNWIND]]:
      // CHECK: landingpad [[LPADTYPE]]
      // CHECK-NEXT: catch
      // CHECK: sync within %[[SYNCREG2]]
      _Cilk_for (int i = 0; i < n; ++i)
        bar(new Foo());
    }
  catch (int e)
    {
      handle_exn(e);
    }
  catch (...)
    {
      handle_exn();
    }
}

// CHECK-LABEL: @_Z27parallelfor_tryblock_inlinei(
void parallelfor_tryblock_inline(int n) {
  // CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
  try
    {
      // CHECK: detach within %[[SYNCREG]], label %[[DETACHED:.+]], label %{{.+}} unwind label %[[DUNWIND:.+]]
      // CHECK: [[DETACHED]]:
      // CHECK: invoke i8* @_Znwm(
      // CHECK: invoke void @_ZN3FooC1Ev(
      // CHECK: invoke i32 @_Z3barP3Foo(
      // CHECK-NEXT: to label %[[INVOKECONT1:.+]] unwind label %[[TASKLPAD:.+]]
      // CHECK: [[TASKLPAD]]:
      // CHECK-NEXT: landingpad [[LPADTYPE:.+]]
      // CHECK-NEXT: catch {{.+}} bitcast
      // CHECK: br i1 {{.+}}, label {{.+}}, label %[[CATCHRESUME:.+]]
      // CHECK: [[CATCHRESUME]]:
      // CHECK: invoke void @llvm.detached.rethrow
      // CHECK: (token %[[SYNCREG]], [[LPADTYPE]] {{.+}})
      // CHECK-NEXT: to label {{.+}} unwind label %[[DUNWIND]]
      // CHECK: [[DUNWIND]]:
      // CHECK: landingpad [[LPADTYPE]]
      // CHECK-NEXT: catch
      // CHECK: sync within %[[SYNCREG]]
      _Cilk_for (int i = 0; i < n; ++i)
        foo(new Foo());
    }
  catch (int e)
    {
      handle_exn(e);
    }
  catch (...)
    {
      handle_exn();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// _Cilk_spawn code snippets
////////////////////////////////////////////////////////////////////////////////

// CHECK-LABEL: @_Z14spawn_noexcepti(
// CHECK-NOT: landingpad
// CHECK-NOT: detached.rethrow
void spawn_noexcept(int n) {
  _Cilk_spawn quuz(n);
  quuz(n);
}

// CHECK-LABEL: @_Z12spawn_excepti(
void spawn_except(int n) {
  // CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED:.+]], label %{{.+}} unwind label %[[DUNWIND:.+]]
  // CHECK: [[DETACHED]]:
  // CHECK: invoke i32 @_Z3barP3Foo(
  // CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[TASKLPAD:.+]]
  // CHECK: [[INVOKECONT]]:
  // CHECK-NEXT: reattach within %[[SYNCREG]]
  // CHECK: [[TASKLPAD]]:
  // CHECK-NEXT: landingpad [[LPADTYPE:.+]]
  // CHECK-NEXT: catch {{.+}} null
  // CHECK: invoke void @llvm.detached.rethrow
  // CHECK: (token %[[SYNCREG]], [[LPADTYPE]] {{.+}})
  // CHECK-NEXT: to label {{.+}} unwind label %[[DUNWIND]]
  // CHECK: [[DUNWIND]]:
  // CHECK: landingpad [[LPADTYPE]]
  // CHECK: sync within %[[SYNCREG]]
  _Cilk_spawn bar(new Foo());
  quuz(n);
}

// CHECK-LABEL: @_Z18spawn_throw_inlinei(
void spawn_throw_inline(int n) {
  // CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
  // CHECK: call i8* @_Znwm(
  // CHECK: invoke void @_ZN3FooC1Ev(
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED:.+]], label %{{.+}} unwind label %[[DUNWIND:.+]]
  // CHECK: [[DETACHED]]:
  // CHECK: invoke i32 @_Z3barP3Foo(
  // CHECK-NEXT: to label %[[INVOKECONT1:.+]] unwind label %[[TASKLPAD:.+]]
  // CHECK: [[TASKLPAD]]:
  // CHECK-NEXT: landingpad [[LPADTYPE:.+]]
  // CHECK-NEXT: catch {{.+}} bitcast
  // CHECK: br i1 {{.+}}, label {{.+}}, label %[[CATCHRESUME:.+]]
  // CHECK: [[CATCHRESUME]]:
  // CHECK: invoke void @llvm.detached.rethrow
  // CHECK: (token %[[SYNCREG]], [[LPADTYPE]] {{.+}})
  // CHECK-NEXT: to label {{.+}} unwind label %[[DUNWIND]]
  // CHECK: [[DUNWIND]]:
  // CHECK: landingpad [[LPADTYPE]]
  // CHECK: sync within %[[SYNCREG]]
  _Cilk_spawn foo(new Foo());
  quuz(n);
}

// CHECK-LABEL: @_Z14spawn_tryblocki(
void spawn_tryblock(int n) {
  // CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
  try
    {
      // CHECK: detach within %[[SYNCREG]], label %[[DETACHED1:.+]], label %[[CONTINUE1:.+]]
      // CHECK-NOT: unwind
      // CHECK: [[DETACHED1]]:
      // CHECK-NEXT: call i32 @_Z4quuzi(
      // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE1]]
      _Cilk_spawn quuz(n);
      // CHECK: detach within %[[SYNCREG]], label %[[DETACHED2:.+]], label %[[CONTINUE2:.+]] unwind label %[[DUNWIND:.+]]
      // CHECK: [[DETACHED2]]:
      // CHECK: invoke i32 @_Z3barP3Foo(
      // CHECK-NEXT: to label %[[INVOKECONT1:.+]] unwind label %[[TASKLPAD:.+]]
      // CHECK: [[INVOKECONT1]]:
      // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE2]]
      _Cilk_spawn bar(new Foo());
      // CHECK: detach within %[[SYNCREG]], label %[[DETACHED3:.+]], label %[[CONTINUE3:.+]]
      // CHECK: [[DETACHED3]]:
      // CHECK-NEXT: call i32 @_Z4quuzi(
      // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE3]]
      _Cilk_spawn quuz(n);
      // CHECK: [[CONTINUE3]]:
      // CHECK: invoke i32 @_Z3barP3Foo(
      // CHECK-NEXT: to label %[[INVOKECONT2:.+]] unwind label %[[CONT3UNWIND:.+]]
      bar(new Foo());
      // CHECK: [[INVOKECONT2]]:
      // CHECK-NEXT: sync within %[[SYNCREG]]
      _Cilk_sync;
    }
  // CHECK-DAG: [[TASKLPAD]]:
  // CHECK-NEXT: landingpad [[LPADTYPE:.+]]
  // CHECK-NEXT: catch {{.+}} null
  // CHECK: invoke void @llvm.detached.rethrow
  // CHECK: (token %[[SYNCREG]], [[LPADTYPE]] {{.+}})
  // CHECK-NEXT: to label {{.+}} unwind label %[[DUNWIND]]
  // CHECK: [[DUNWIND]]:
  // CHECK: landingpad [[LPADTYPE]]
  // CHECK-NEXT: catch
  // CHECK: [[CONT3UNWIND]]:
  // CHECK: landingpad [[LPADTYPE]]
  // CHECK-NEXT: catch
  // CHECK: sync within %[[SYNCREG]]
  catch (int e)
    {
      handle_exn(e);
    }
  catch (...)
    {
      handle_exn();
    }
}
