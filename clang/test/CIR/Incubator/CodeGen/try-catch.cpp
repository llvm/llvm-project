// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir-flat %s -o %t.cir.flat
// RUN: FileCheck --check-prefix=FLAT --input-file=%t.cir.flat %s

double division(int a, int b);

// CHECK: cir.func {{.*}} @_Z2tcv()
unsigned long long tc() {
  int x = 50, y = 3;
  unsigned long long z;

  try {
    // CHECK: cir.scope {
    // CHECK: %[[local_a:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
    // CHECK: %[[msg:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["msg"]
    // CHECK: %[[idx:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["idx"]

    // CHECK: cir.try {
    int a = 4;
    z = division(x, y);
    // CHECK: %[[div_res:.*]] = cir.call exception @_Z8divisionii({{.*}}) : (!s32i, !s32i) -> !cir.double
    a++;

  } catch (int idx) {
    // CHECK: } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i> {
    // CHECK:   %[[catch_idx_addr:.*]] = cir.catch_param -> !cir.ptr<!s32i>
    // CHECK:   %[[idx_load:.*]] = cir.load{{.*}} %[[catch_idx_addr]] : !cir.ptr<!s32i>, !s32i
    // CHECK:   cir.store{{.*}} %[[idx_load]], %[[idx]] : !s32i, !cir.ptr<!s32i>
    z = 98;
    idx++;
  } catch (const char* msg) {
    // CHECK: }, type #cir.global_view<@_ZTIPKc> : !cir.ptr<!u8i> {
    // CHECK:   %[[msg_addr:.*]] = cir.catch_param -> !cir.ptr<!s8i>
    // CHECK:   cir.store{{.*}} %[[msg_addr]], %[[msg]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
    z = 99;
    (void)msg[0];
  } // CHECK: }, #cir.unwind {
    // CHECK: cir.resume
    // CHECK-NEXT: }

  return z;
}

// CHECK: cir.func {{.*}} @_Z3tc2v
unsigned long long tc2() {
  int x = 50, y = 3;
  unsigned long long z;

  try {
    int a = 4;
    z = division(x, y);
    a++;
  } catch (int idx) {
    z = 98;
    idx++;
  } catch (const char* msg) {
    z = 99;
    (void)msg[0];
  } catch (...) {
    // CHECK: }, type #cir.all {
    // CHECK:   cir.catch_param
    // CHECK:   cir.const #cir.int<100> : !s32i
    z = 100;
  }

  return z;
}

// CHECK: cir.func {{.*}} @_Z3tc3v
unsigned long long tc3() {
  int x = 50, y = 3;
  unsigned long long z;

  try {
    z = division(x, y);
  } catch (...) {
    // CHECK: } catch [type #cir.all {
    // CHECK:   cir.catch_param
    // CHECK:   cir.const #cir.int<100> : !s32i
    z = 100;
  }

  return z;
}

// CHECK: cir.func {{.*}} @_Z3tc4v()
unsigned long long tc4() {
  int x = 50, y = 3;
  unsigned long long z;

  // CHECK-NOT: cir.try
  try {
    int a = 4;
    a++;

    // CHECK: cir.scope {
    // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
    // CHECK-NOT: cir.alloca !cir.ptr<!cir.eh.info>
    // CHECK: cir.const #cir.int<4> : !s32i
    // CHECK: cir.unary(inc,
    // CHECK: cir.store{{.*}} %11, %8 : !s32i, !cir.ptr<!s32i>
  } catch (int idx) {
    z = 98;
    idx++;
  }

  return z;
}

struct S {
  S() {};
  int a;
};

// CHECK: cir.func {{.*}} @_Z3tc5v()
void tc5() {
  try {
    S s;
  } catch (...) {
    tc5();
  }
}

// CHECK: cir.try {
// CHECK: cir.call exception @_ZN1SC2Ev({{.*}}) : (!cir.ptr<!rec_S>) -> ()
// CHECK: cir.yield
// CHECK: } catch [type #cir.all {
// CHECK:  {{.*}} = cir.catch_param -> !cir.ptr<!void>
// CHECK:  cir.call exception @_Z3tc5v() : () -> ()
// CHECK:  cir.yield
// CHECK: }]

// CHECK: cir.func {{.*}} @_Z3tc6v()
void tc6() {
  int r = 1;
  try {
    return;
    ++r;
  } catch (...) {
  }
}

// CHECK: cir.scope {
// CHECK:   cir.try {
// CHECK:     cir.return
// CHECK:   ^bb1:  // no predecessors
// CHECK:     %[[V2:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[V3:.*]] = cir.unary(inc, %[[V2]]) nsw : !s32i, !s32i
// CHECK:     cir.store{{.*}} %[[V3]], {{.*}} : !s32i, !cir.ptr<!s32i>
// CHECK:     cir.yield
// CHECK:   }
// CHECK: }

// CHECK: cir.func {{.*}} @_Z3tc7v()
void tc7() {
  int r = 1;
  try {
    ++r;
    return;
  } catch (...) {
  }
}

// CHECK: cir.scope {
// CHECK:   cir.try {
// CHECK:     %[[V2:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[V3:.*]] = cir.unary(inc, %[[V2]]) nsw : !s32i, !s32i
// CHECK:     cir.store{{.*}} %[[V3]], {{.*}} : !s32i, !cir.ptr<!s32i>
// CHECK:     cir.return
// CHECK:   }
// CHECK: }

struct S2 {
  int a, b;
};

void tc8() {
  try {
    S2 s{1, 2};
  } catch (...) {
  }
}

// CHECK: cir.scope {
// CHECK:   %[[V0:.*]] = cir.alloca !rec_S2, !cir.ptr<!rec_S2>, ["s", init] {alignment = 4 : i64}
// CHECK:   cir.try {
// CHECK:     %[[V1:.*]] = cir.const #cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s32i}> : !rec_S2
// CHECK:     cir.store align(4) %[[V1]], %[[V0]] : !rec_S2, !cir.ptr<!rec_S2>
// CHECK:     cir.yield
// CHECK:   }
// CHECK: }

// FLAT: cir.func {{.*}} @_Z3tc8v()
// FLAT:   %[[V0:.*]] = cir.alloca !rec_S2, !cir.ptr<!rec_S2>, ["s", init] {alignment = 4 : i64}
// FLAT:   cir.br ^bb[[#B1:]]
// FLAT: ^bb[[#B1]]:
// FLAT:   cir.br ^bb[[#B2:]]
// FLAT: ^bb[[#B2]]:
// FLAT:   %[[V1:.*]] = cir.const #cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s32i}> : !rec_S2
// FLAT:   cir.store align(4) %[[V1]], %[[V0]] : !rec_S2, !cir.ptr<!rec_S2>
// FLAT:   cir.br ^bb[[#B3:]]
// FLAT: ^bb[[#B3]]:
// FLAT:   cir.br ^bb[[#B4:]]
// FLAT: ^bb[[#B4]]:
// FLAT:   cir.return
// FLAT: }
