// Tests that we assign appropriate identifiers to indirect calls and targets.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,MS %s

// CHECK-DAG: define {{(dso_local)?}} void @foo({{.*}} !type [[F_TVOID:![0-9]+]]
void foo() {
}

// CHECK-DAG: define {{(dso_local)?}} void @bar({{.*}} !type [[F_TVOID]]
void bar() {
  void (*fp)() = foo;
  // CHECK: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  fp();
}

// CHECK-DAG: define {{(dso_local)?}} i32 @baz({{.*}} !type [[F_TPRIMITIVE:![0-9]+]]
int baz(char a, float b, double c) {
  return 1;
}

// CHECK-DAG: define {{(dso_local)?}} ptr @qux({{.*}} !type [[F_TPTR:![0-9]+]]
int *qux(char *a, float *b, double *c) {
  return 0;
}

// CHECK-DAG: define {{(dso_local)?}} void @corge({{.*}} !type [[F_TVOID]]
void corge() {
  int (*fp_baz)(char, float, double) = baz;
  // CHECK: call i32 {{.*}}, !callee_type [[F_TPRIMITIVE_CT:![0-9]+]]  
  fp_baz('a', .0f, .0);

  int *(*fp_qux)(char *, float *, double *) = qux;
  // CHECK: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  fp_qux(0, 0, 0);
}

struct st1 {
  int *(*fp)(char *, float *, double *);
};

struct st2 {
  struct st1 m;
};

// CHECK-DAG: define {{(dso_local)?}} void @stparam({{.*}} !type [[F_TSTRUCT:![0-9]+]]
void stparam(struct st2 a, struct st2 *b) {}

// CHECK-DAG: define {{(dso_local)?}} void @stf({{.*}} !type [[F_TVOID]]
void stf() {
  struct st1 St1;
  St1.fp = qux;
  // CHECK: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]  
  St1.fp(0, 0, 0);

  struct st2 St2;
  St2.m.fp = qux;
  // CHECK: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  St2.m.fp(0, 0, 0);

  // CHECK: call void {{.*}}, !callee_type [[F_TSTRUCT_CT:![0-9]+]]
  void (*fp_stparam)(struct st2, struct st2 *) = stparam;
  fp_stparam(St2, &St2);
}

// CHECK-DAG: [[F_TVOID_CT]] = !{[[F_TVOID:![0-9]+]]}
// ITANIUM-DAG: [[F_TVOID]] = !{i64 0, !"_ZTSFvE.generalized"}
// MS-DAG:  [[F_TVOID]] = !{i64 0, !"?6AX@Z.generalized"}

// CHECK-DAG: [[F_TPRIMITIVE_CT]] = !{[[F_TPRIMITIVE:![0-9]+]]}
// ITANIUM-DAG: [[F_TPRIMITIVE]] = !{i64 0, !"_ZTSFicfdE.generalized"}
// MS-DAG:      [[F_TPRIMITIVE]] = !{i64 0, !"?6AHDMN@Z.generalized"}

// CHECK-DAG: [[F_TPTR_CT]] = !{[[F_TPTR:![0-9]+]]}
// ITANIUM-DAG: [[F_TPTR]] = !{i64 0, !"_ZTSFPvS_S_S_E.generalized"}
// MS-DAG:      [[F_TPTR]] = !{i64 0, !"?6APEAXPEAX00@Z.generalized"}

// CHECK-DAG: [[F_TSTRUCT_CT]] = !{[[F_TSTRUCT:![0-9]+]]}
// ITANIUM-DAG: [[F_TSTRUCT]] = !{i64 0, !"_ZTSFv3st2PvE.generalized"}
// MS-DAG:      [[F_TSTRUCT]] = !{i64 0, !"?6AXUst2@@PEAX@Z.generalized"}
