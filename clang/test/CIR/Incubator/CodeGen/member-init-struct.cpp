// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

struct A {int a;};
struct B {float a;};
struct C {
  union {
    A a;
    B b[10];
  };
  int c;
  int d[10];
  void (C::*e)();
  C() : a(), c(), d(), e() {}
  C(A x) : a(x) {}
  C(void (C::*x)(), int y) : b(), c(y), e(x) {}
};

// CHECK-LABEL:   cir.global external @x = #cir.zero : !rec_A
A x;
C a, b(x), c(0, 2);

// CHECK-LABEL: @_ZN1CC2Ev
// CHECK:   %[[VAL_1:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["this", init] {alignment = 8 : i64}
// CHECK:   cir.store{{.*}} %{{.*}}, %[[VAL_1]] : !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>
// CHECK:   %[[VAL_2:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
// CHECK:   %[[VAL_3:.*]] = cir.get_member %[[VAL_2]][0] {name = ""} : !cir.ptr<!rec_C> -> !cir.ptr<!rec_anon2E0>
// CHECK:   %[[VAL_4:.*]] = cir.get_member %[[VAL_3]][0] {name = "a"} : !cir.ptr<!rec_anon2E0> -> !cir.ptr<!rec_A>
// CHECK:   %[[VAL_5:.*]] = cir.const {{.*}} : !rec_A
// CHECK:   cir.store{{.*}} %[[VAL_5]], %[[VAL_4]] : !rec_A, !cir.ptr<!rec_A>
// Trivial default constructor call is lowered away.
// CHECK:   %[[VAL_6:.*]] = cir.get_member %[[VAL_2]][1] {name = "c"} : !cir.ptr<!rec_C> -> !cir.ptr<!s32i>
// CHECK:   %[[VAL_7:.*]] = cir.const {{.*}}<0> : !s32i
// CHECK:   cir.store{{.*}} %[[VAL_7]], %[[VAL_6]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[VAL_8:.*]] = cir.get_member %[[VAL_2]][2] {name = "d"} : !cir.ptr<!rec_C> -> !cir.ptr<!cir.array<!s32i x 10>>
// CHECK:   %[[VAL_9:.*]] = cir.const {{.*}} : !cir.array<!s32i x 10>
// CHECK:   cir.store{{.*}} %[[VAL_9]], %[[VAL_8]] : !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>
// CHECK:   %[[VAL_10:.*]] = cir.get_member %[[VAL_2]][4] {name = "e"} : !cir.ptr<!rec_C> -> !cir.ptr<!cir.method<!cir.func<()> in !rec_C>>
// CHECK:   %[[VAL_11:.*]] = cir.const #cir.method<null> : !cir.method<!cir.func<()> in !rec_C>
// CHECK:   cir.store{{.*}} %[[VAL_11]], %[[VAL_10]] : !cir.method<!cir.func<()> in !rec_C>, !cir.ptr<!cir.method<!cir.func<()> in !rec_C>>
// CHECK:   cir.return
