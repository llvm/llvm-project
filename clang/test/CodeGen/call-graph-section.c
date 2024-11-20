// Tests that we assign appropriate identifiers to indirect calls and targets.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fcall-graph-section -S \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fcall-graph-section -S \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,MS %s

// CHECK-DAG: define {{(dso_local)?}} void @foo({{.*}} !type [[F_TVOID:![0-9]+]]
void foo() {
}

// CHECK-DAG: define {{(dso_local)?}} void @bar({{.*}} !type [[F_TVOID]]
void bar() {
  void (*fp)() = foo;
  // ITANIUM: call {{.*}} [ "type"(metadata !"_ZTSFvE.generalized") ]
  // MS:      call {{.*}} [ "type"(metadata !"?6AX@Z.generalized") ]
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
  // ITANIUM: call i32 {{.*}} [ "type"(metadata !"_ZTSFicfdE.generalized") ]
  // MS:      call i32 {{.*}} [ "type"(metadata !"?6AHDMN@Z.generalized") ]
  fp_baz('a', .0f, .0);

  int *(*fp_qux)(char *, float *, double *) = qux;
  // ITANIUM: call ptr {{.*}} [ "type"(metadata !"_ZTSFPvS_S_S_E.generalized") ]
  // MS:      call ptr {{.*}} [ "type"(metadata !"?6APEAXPEAX00@Z.generalized") ]
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
  // ITANIUM: call ptr {{.*}} [ "type"(metadata !"_ZTSFPvS_S_S_E.generalized") ]
  // MS:      call ptr {{.*}} [ "type"(metadata !"?6APEAXPEAX00@Z.generalized") ]
  St1.fp(0, 0, 0);

  struct st2 St2;
  St2.m.fp = qux;
  // ITANIUM: call ptr {{.*}} [ "type"(metadata !"_ZTSFPvS_S_S_E.generalized") ]
  // MS:      call ptr {{.*}} [ "type"(metadata !"?6APEAXPEAX00@Z.generalized") ]
  St2.m.fp(0, 0, 0);

  // ITANIUM: call void {{.*}} [ "type"(metadata !"_ZTSFv3st2PvE.generalized") ]
  // MS:      call void {{.*}} [ "type"(metadata !"?6AXUst2@@PEAX@Z.generalized") ]
  void (*fp_stparam)(struct st2, struct st2 *) = stparam;
  fp_stparam(St2, &St2);
}

// ITANIUM-DAG: [[F_TVOID]] = !{i64 0, !"_ZTSFvE.generalized"}
// MS-DAG:      [[F_TVOID]] = !{i64 0, !"?6AX@Z.generalized"}

// ITANIUM-DAG: [[F_TPRIMITIVE]] = !{i64 0, !"_ZTSFicfdE.generalized"}
// MS-DAG:      [[F_TPRIMITIVE]] = !{i64 0, !"?6AHDMN@Z.generalized"}

// ITANIUM-DAG: [[F_TPTR]] = !{i64 0, !"_ZTSFPvS_S_S_E.generalized"}
// MS-DAG:      [[F_TPTR]] = !{i64 0, !"?6APEAXPEAX00@Z.generalized"}

// ITANIUM-DAG: [[F_TSTRUCT]] = !{i64 0, !"_ZTSFv3st2PvE.generalized"}
// MS-DAG:      [[F_TSTRUCT]] = !{i64 0, !"?6AXUst2@@PEAX@Z.generalized"}
