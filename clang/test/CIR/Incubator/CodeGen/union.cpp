// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct { int x; } yolo;
typedef union { yolo y; struct { int lifecnt; }; } yolm;
typedef union { yolo y; struct { int *lifecnt; int genpad; }; } yolm2;
typedef union { yolo y; struct { bool life; int genpad; }; } yolm3;

// CHECK-DAG: !rec_U23A3ADummy = !cir.record<struct "U2::Dummy" {!s16i, !cir.float} #cir.record.decl.ast>
// CHECK-DAG: !rec_anon2E0 = !cir.record<struct "anon.0" {!s32i} #cir.record.decl.ast>
// CHECK-DAG: !rec_anon2E2 = !cir.record<struct "anon.2" {!cir.bool, !s32i} #cir.record.decl.ast>
// CHECK-DAG: !rec_yolo = !cir.record<struct "yolo" {!s32i} #cir.record.decl.ast>
// CHECK-DAG: !rec_anon2E1 = !cir.record<struct "anon.1" {!cir.ptr<!s32i>, !s32i} #cir.record.decl.ast>

// CHECK-DAG: !rec_yolm = !cir.record<union "yolm" {!rec_yolo, !rec_anon2E0}>
// CHECK-DAG: !rec_yolm3 = !cir.record<union "yolm3" {!rec_yolo, !rec_anon2E2}>
// CHECK-DAG: !rec_yolm2 = !cir.record<union "yolm2" {!rec_yolo, !rec_anon2E1}>

// Should generate a union type with all members preserved.
union U {
  bool b;
  short s;
  int i;
  float f;
  double d;
};
// CHECK-DAG: !rec_U = !cir.record<union "U" {!cir.bool, !s16i, !s32i, !cir.float, !cir.double}>

// Should generate unions with complex members.
union U2 {
  bool b;
  struct Dummy {
    short s;
    float f;
  } s;
} u2;
// CHECK-DAG: !cir.record<union "U2" {!cir.bool, !rec_U23A3ADummy} #cir.record.decl.ast>

// Should genereate unions without padding.
union U3 {
  short b;
  U u;
} u3;
// CHECK-DAG: !rec_U3 = !cir.record<union "U3" {!s16i, !rec_U} #cir.record.decl.ast>

void m() {
  yolm q;
  yolm2 q2;
  yolm3 q3;
}

// CHECK:   cir.func {{.*}} @_Z1mv()
// CHECK:   cir.alloca !rec_yolm, !cir.ptr<!rec_yolm>, ["q"] {alignment = 4 : i64}
// CHECK:   cir.alloca !rec_yolm2, !cir.ptr<!rec_yolm2>, ["q2"] {alignment = 8 : i64}
// CHECK:   cir.alloca !rec_yolm3, !cir.ptr<!rec_yolm3>, ["q3"] {alignment = 4 : i64}

void shouldGenerateUnionAccess(union U u) {
  u.b = true;
  // CHECK: %[[#BASE:]] = cir.get_member %0[0] {name = "b"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.bool>
  // CHECK: cir.store{{.*}} %{{.+}}, %[[#BASE]] : !cir.bool, !cir.ptr<!cir.bool>
  u.b;
  // CHECK: cir.get_member %0[0] {name = "b"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.bool>
  u.i = 1;
  // CHECK: %[[#BASE:]] = cir.get_member %0[2] {name = "i"} : !cir.ptr<!rec_U> -> !cir.ptr<!s32i>
  // CHECK: cir.store{{.*}} %{{.+}}, %[[#BASE]] : !s32i, !cir.ptr<!s32i>
  u.i;
  // CHECK: %[[#BASE:]] = cir.get_member %0[2] {name = "i"} : !cir.ptr<!rec_U> -> !cir.ptr<!s32i>
  u.f = 0.1F;
  // CHECK: %[[#BASE:]] = cir.get_member %0[3] {name = "f"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.float>
  // CHECK: cir.store{{.*}} %{{.+}}, %[[#BASE]] : !cir.float, !cir.ptr<!cir.float>
  u.f;
  // CHECK: %[[#BASE:]] = cir.get_member %0[3] {name = "f"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.float>
  u.d = 0.1;
  // CHECK: %[[#BASE:]] = cir.get_member %0[4] {name = "d"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.double>
  // CHECK: cir.store{{.*}} %{{.+}}, %[[#BASE]] : !cir.double, !cir.ptr<!cir.double>
  u.d;
  // CHECK: %[[#BASE:]] = cir.get_member %0[4] {name = "d"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.double>
}

typedef union {
  short a;
  int b;
} A;

void noCrushOnDifferentSizes() {
  A a = {0};
  // CHECK:  %[[#TMP0:]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a"] {alignment = 4 : i64}
  // CHECK:  %[[#TMP1:]] = cir.cast bitcast %[[#TMP0]] : !cir.ptr<!rec_A> -> !cir.ptr<!rec_anon_struct>
  // CHECK:  %[[#TMP2:]] = cir.const #cir.zero : !rec_anon_struct
  // CHECK:  cir.store{{.*}} %[[#TMP2]], %[[#TMP1]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
}
