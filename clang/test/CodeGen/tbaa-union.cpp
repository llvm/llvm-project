// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s
//
// Check that we generate correct TBAA information for accesses to union
// members.

struct X {
  int a, b;
  int arr[3];
  int c, d;
};

union U {
  int i;
  X x;
  int j;
};

struct S {
  U u, v;
};

union N {
  int i;
  S s;
  int j;
};

struct R {
  N n, m;
};

int f1(U *p) {
// CHECK-LABEL: _Z2f1P1U
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_U_j:!.*]]
  return p->j;
}

int f2(S *p) {
// CHECK-LABEL: _Z2f2P1S
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_S_u_i:!.*]]
  return p->u.i;
}

int f3(S *p) {
// CHECK-LABEL: _Z2f3P1S
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_S_v_j:!.*]]
  return p->v.j;
}

int f4(S *p) {
// CHECK-LABEL: _Z2f4P1S
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_S_u_x_b:!.*]]
  return p->u.x.b;
}

int f5(S *p) {
// CHECK-LABEL: _Z2f5P1S
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_S_v_x_b:!.*]]
  return p->v.x.b;
}

int f6(S *p) {
// CHECK-LABEL: _Z2f6P1S
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_S_u_x_arr:!.*]]
  return p->u.x.arr[1];
}

int f7(S *p) {
// CHECK-LABEL: _Z2f7P1S
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_S_v_x_arr:!.*]]
  return p->v.x.arr[1];
}

int f8(N *p) {
// CHECK-LABEL: _Z2f8P1N
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_N_s_v_x_c:!.*]]
  return p->s.v.x.c;
}

int f9(R *p) {
// CHECK-LABEL: _Z2f9P1R
// CHECK: load i32, i32* {{.*}}, !tbaa [[TAG_R_m_s_v_x_c:!.*]]
  return p->m.s.v.x.c;
}

// CHECK-DAG: [[TAG_U_j]] = !{[[TYPE_U:!.*]], [[TYPE_union_member:!.*]], i64 0}
// CHECK-DAG: [[TAG_S_u_i]] = !{[[TYPE_S:!.*]], [[TYPE_union_member]], i64 0}
// CHECK-DAG: [[TAG_S_u_x_b]] = !{[[TYPE_S:!.*]], [[TYPE_union_member]], i64 0}
// CHECK-DAG: [[TAG_S_u_x_arr]] = !{[[TYPE_S:!.*]], [[TYPE_union_member]], i64 0}
// CHECK-DAG: [[TAG_S_v_j]] = !{[[TYPE_S:!.*]], [[TYPE_union_member]], i64 28}
// CHECK-DAG: [[TAG_S_v_x_b]] = !{[[TYPE_S:!.*]], [[TYPE_union_member]], i64 28}
// CHECK-DAG: [[TAG_S_v_x_arr]] = !{[[TYPE_S:!.*]], [[TYPE_union_member]], i64 28}
// CHECK-DAG: [[TAG_N_s_v_x_c]] = !{[[TYPE_N:!.*]], [[TYPE_union_member]], i64 0}
// CHECK-DAG: [[TAG_R_m_s_v_x_c]] = !{[[TYPE_R:!.*]], [[TYPE_union_member]], i64 56}
// CHECK-DAG: [[TYPE_U]] = !{!"_ZTS1U", [[TYPE_union_member]], i64 0}
// CHECK-DAG: [[TYPE_S]] = !{!"_ZTS1S", [[TYPE_U]], i64 0, [[TYPE_U]], i64 28}
// CHECK-DAG: [[TYPE_N]] = !{!"_ZTS1N", [[TYPE_union_member]], i64 0}
// CHECK-DAG: [[TYPE_R]] = !{!"_ZTS1R", [[TYPE_N]], i64 0, [[TYPE_N]], i64 56}
// CHECK-DAG: [[TYPE_union_member]] = !{!"union member", [[TYPE_char:!.*]], i64 0}
// CHECK-DAG: [[TYPE_char]] = !{!"omnipotent char", {{.*}}, i64 0}
