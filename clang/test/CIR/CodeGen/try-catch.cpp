// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

double division(int a, int b);

// CHECK: cir.func @_Z2tcv()
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
    // CHECK:   %[[idx_load:.*]] = cir.load %[[catch_idx_addr]] : !cir.ptr<!s32i>, !s32i
    // CHECK:   cir.store %[[idx_load]], %[[idx]] : !s32i, !cir.ptr<!s32i>
    z = 98;
    idx++;
  } catch (const char* msg) {
    // CHECK: }, type #cir.global_view<@_ZTIPKc> : !cir.ptr<!u8i> {
    // CHECK:   %[[msg_addr:.*]] = cir.catch_param -> !cir.ptr<!s8i>
    // CHECK:   cir.store %[[msg_addr]], %[[msg]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
    z = 99;
    (void)msg[0];
  } // CHECK: }, #cir.unwind {
    // CHECK: cir.resume
    // CHECK-NEXT: }

  return z;
}

// CHECK: cir.func @_Z3tc2v
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

// CHECK: cir.func @_Z3tc3v
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

// CHECK: cir.func @_Z3tc4v()
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
    // CHECK: cir.store %11, %8 : !s32i, !cir.ptr<!s32i>
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

// CHECK: cir.func @_Z3tc5v()
void tc5() {
  try {
    S s;
  } catch (...) {
    tc5();
  }
}

// CHECK: cir.try {
// CHECK: cir.call exception @_ZN1SC2Ev({{.*}}) : (!cir.ptr<!ty_S>) -> ()
// CHECK: cir.yield
// CHECK: } catch [type #cir.all {
// CHECK:  {{.*}} = cir.catch_param -> !cir.ptr<!void>
// CHECK:  cir.call exception @_Z3tc5v() : () -> ()
// CHECK:  cir.yield
// CHECK: }]

// CHECK: cir.func @_Z3tc6v()
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
// CHECK:     %[[V2:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK:     %[[V3:.*]] = cir.unary(inc, %[[V2]]) : !s32i, !s32i
// CHECK:     cir.store %[[V3]], {{.*}} : !s32i, !cir.ptr<!s32i>
// CHECK:     cir.yield
// CHECK:   }
// CHECK: }
