// RUN: %clang_cc1 -emit-llvm -o - %s -triple x86_64-linux | FileCheck %s

struct A {
  A();
  A(const A &);
  ~A();
  operator bool();
  void *data;
};

A make();
bool cond();
void f(int);

// PR49585: Ensure that 'continue' performs the proper cleanups in the presence
// of a for loop condition variable.
//
// CHECK: define {{.*}} void @_Z7PR49585v(
void PR49585() {
  for (
      // CHECK: call void @_Z1fi(i32 1)
      // CHECK: br label %[[for_cond:.*]]
      f(1);

      // CHECK: [[for_cond]]:
      // CHECK: call {{.*}} @_Z4makev(
      // CHECK: call {{.*}} @_ZN1AcvbEv(
      // CHECK: br i1 {{.*}}, label %[[for_body:.*]], label %[[for_cond_cleanup:.*]]
      A a = make();

      // CHECK: [[for_cond_cleanup]]:
      // CHECK: store
      // CHECK: br label %[[cleanup:.*]]

      f(2)) {
    // CHECK: [[for_body]]:
    // CHECK: call {{.*}} @_Z4condv(
    // CHECK: br i1 {{.*}}, label %[[if_then:.*]], label %[[if_end:.*]]
    if (cond()) {
      // CHECK: [[if_then]]:
      // CHECK: call {{.*}} @_Z1fi(i32 3)
      // CHECK: br label %[[for_inc:.*]]
      f(3);
      continue;
    }

    // CHECK: [[if_end]]:
    // CHECK: call {{.*}} @_Z1fi(i32 4)
    // CHECK: br label %[[for_inc]]
    f(4);
  }

  // CHECK: [[for_inc]]:
  // CHECK: call void @_Z1fi(i32 2)
  // CHECK: store
  // CHECK: br label %[[cleanup]]

  // CHECK: [[cleanup]]:
  // CHECK: call void @_ZN1AD1Ev(
  // CHECK: load
  // CHECK: switch {{.*}} label
  // CHECK-NEXT: label %[[cleanup_cont:.*]]
  // CHECK-NEXT: label %[[for_end:.*]]

  // CHECK: [[cleanup_cont]]:
  // CHECK: br label %[[for_cond]]

  // CHECK [[for_end]]:
  // CHECK: ret void
}

// CHECK: define {{.*}} void @_Z13PR49585_breakv(
void PR49585_break() {
  for (
      // CHECK: call void @_Z1fi(i32 1)
      // CHECK: br label %[[for_cond:.*]]
      f(1);

      // CHECK: [[for_cond]]:
      // CHECK: call {{.*}} @_Z4makev(
      // CHECK: call {{.*}} @_ZN1AcvbEv(
      // CHECK: br i1 {{.*}}, label %[[for_body:.*]], label %[[for_cond_cleanup:.*]]
      A a = make();

      // CHECK: [[for_cond_cleanup]]:
      // CHECK: store
      // CHECK: br label %[[cleanup:.*]]

      f(2)) {
    // CHECK: [[for_body]]:
    // CHECK: call {{.*}} @_Z4condv(
    // CHECK: br i1 {{.*}}, label %[[if_then:.*]], label %[[if_end:.*]]
    if (cond()) {
      // CHECK: [[if_then]]:
      // CHECK: call {{.*}} @_Z1fi(i32 3)
      // CHECK: store
      // CHECK: br label %[[cleanup:.*]]
      f(3);
      break;
    }

    // CHECK: [[if_end]]:
    // CHECK: call {{.*}} @_Z1fi(i32 4)
    // CHECK: br label %[[for_inc]]
    f(4);
  }

  // CHECK: [[for_inc]]:
  // CHECK: call void @_Z1fi(i32 2)
  // CHECK: store
  // CHECK: br label %[[cleanup]]

  // CHECK: [[cleanup]]:
  // CHECK: call void @_ZN1AD1Ev(
  // CHECK: load
  // CHECK: switch {{.*}} label
  // CHECK-NEXT: label %[[cleanup_cont:.*]]
  // CHECK-NEXT: label %[[for_end:.*]]

  // CHECK: [[cleanup_cont]]:
  // CHECK: br label %[[for_cond]]

  // CHECK [[for_end]]:
  // CHECK: ret void
}

// CHECK: define {{.*}} void @_Z16incless_for_loopv(
void incless_for_loop() {
  // CHECK: br label %[[for_cond:.*]]
  // CHECK: [[for_cond]]:
  // CHECK:   br i1 {{.*}}, label %[[for_body:.*]], label %[[for_end:.*]]
  // CHECK: [[for_body]]:
  // CHECK:   br label %[[for_cond]]
  // CHECK: [[for_end]]:
  // CHECK:   ret void
  // CHECK: }
  for (; int b = 0;) continue;
}
