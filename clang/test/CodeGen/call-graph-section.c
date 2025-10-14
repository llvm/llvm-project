// Tests that we assign appropriate identifiers to indirect calls and targets.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=MS %s

// ITANIUM-LABEL: define {{(dso_local)?}} void @foo(
// ITANIUM: {{.*}} !type [[F_TVOID:![0-9]+]]
// MS-LABEL: define {{(dso_local)?}} void @foo(
// MS: {{.*}} !type [[F_TVOID:![0-9]+]]
void foo() {
}

// ITANIUM-LABEL: define {{(dso_local)?}} void @bar(
// ITANIUM: {{.*}} !type [[F_TVOID]]
// MS-LABEL: define {{(dso_local)?}} void @bar(
// MS: {{.*}} !type [[F_TVOID]]
void bar() {
  void (*fp)() = foo;
  // ITANIUM: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  fp();
}

// ITANIUM-LABEL: define {{(dso_local)?}} i32 @baz(
// ITANIUM: {{.*}} !type [[F_TPRIMITIVE:![0-9]+]]
// MS-LABEL: define {{(dso_local)?}} i32 @baz(
// MS: {{.*}} !type [[F_TPRIMITIVE:![0-9]+]]
int baz(char a, float b, double c) {
  return 1;
}

// ITANIUM-LABEL: define {{(dso_local)?}} ptr @qux(
// ITANIUM: {{.*}} !type [[F_TPTR:![0-9]+]]
// MS-LABEL: define {{(dso_local)?}} ptr @qux(
// MS: {{.*}} !type [[F_TPTR:![0-9]+]]
int *qux(char *a, float *b, double *c) {
  return 0;
}

// ITANIUM-LABEL: define {{(dso_local)?}} void @corge(
// ITANIUM: {{.*}} !type [[F_TVOID]]
// MS-LABEL: define {{(dso_local)?}} void @corge(
// MS: {{.*}} !type [[F_TVOID]]
void corge() {
  int (*fp_baz)(char, float, double) = baz;
  // ITANIUM: call i32 {{.*}}, !callee_type [[F_TPRIMITIVE_CT:![0-9]+]]
  // MS: call i32 {{.*}}, !callee_type [[F_TPRIMITIVE_CT:![0-9]+]]
  fp_baz('a', .0f, .0);

  int *(*fp_qux)(char *, float *, double *) = qux;
  // ITANIUM: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  // MS: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  fp_qux(0, 0, 0);
}

struct st1 {
  int *(*fp)(char *, float *, double *);
};

struct st2 {
  struct st1 m;
};

// ITANIUM-LABEL: define {{(dso_local)?}} void @stparam(
// ITANIUM: {{.*}} !type [[F_TSTRUCT:![0-9]+]]
// MS-LABEL: define {{(dso_local)?}} void @stparam(
// MS: {{.*}} !type [[F_TSTRUCT:![0-9]+]]
void stparam(struct st2 a, struct st2 *b) {}

// ITANIUM-LABEL: define {{(dso_local)?}} void @stf(
// ITANIUM: {{.*}} !type [[F_TVOID]]
// MS-LABEL: define {{(dso_local)?}} void @stf(
// MS: {{.*}} !type [[F_TVOID]]
void stf() {
  struct st1 St1;
  St1.fp = qux;
  // ITANIUM: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  // MS: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  St1.fp(0, 0, 0);

  struct st2 St2;
  St2.m.fp = qux;
  // ITANIUM: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  // MS: call ptr {{.*}}, !callee_type [[F_TPTR_CT:![0-9]+]]
  St2.m.fp(0, 0, 0);

  // ITANIUM: call void {{.*}}, !callee_type [[F_TSTRUCT_CT:![0-9]+]]
  // MS: call void {{.*}}, !callee_type [[F_TSTRUCT_CT:![0-9]+]]
  void (*fp_stparam)(struct st2, struct st2 *) = stparam;
  fp_stparam(St2, &St2);
}

// ITANIUM: [[F_TVOID]] = !{i64 0, !"_ZTSFvE.generalized"}
// ITANIUM: [[F_TVOID_CT]] = !{[[F_TVOID:![0-9]+]]}
// ITANIUM: [[F_TPRIMITIVE]] = !{i64 0, !"_ZTSFicfdE.generalized"}
// ITANIUM: [[F_TPTR]] = !{i64 0, !"_ZTSFPiPcPfPdE.generalized"}
// ITANIUM: [[F_TPRIMITIVE_CT]] = !{[[F_TPRIMITIVE:![0-9]+]]}
// ITANIUM: [[F_TPTR_CT]] = !{[[F_TPTR:![0-9]+]]}
// ITANIUM: [[F_TSTRUCT]] = !{i64 0, !"_ZTSFv3st2PS_E.generalized"}
// ITANIUM: [[F_TSTRUCT_CT]] = !{[[F_TSTRUCT:![0-9]+]]}

// MS: [[F_TVOID]] = !{i64 0, !"?6AX@Z.generalized"}
// MS: [[F_TVOID_CT]] = !{[[F_TVOID:![0-9]+]]}
// MS: [[F_TPRIMITIVE]] = !{i64 0, !"?6AHDMN@Z.generalized"}
// MS: [[F_TPTR]] = !{i64 0, !"?6APEAHPEADPEAMPEAN@Z.generalized"}
// MS: [[F_TPRIMITIVE_CT]] = !{[[F_TPRIMITIVE:![0-9]+]]}
// MS: [[F_TPTR_CT]] = !{[[F_TPTR:![0-9]+]]}
// MS: [[F_TSTRUCT]] = !{i64 0, !"?6AXUst2@@PEAU0@@Z.generalized"}
// MS: [[F_TSTRUCT_CT]] = !{[[F_TSTRUCT:![0-9]+]]}
