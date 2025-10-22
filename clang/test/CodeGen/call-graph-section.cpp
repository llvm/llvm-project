// Tests that we assign appropriate identifiers to indirect calls and targets
// specifically for C++ class and instance methods.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=FT %s < %t
// RUN: FileCheck --check-prefix=CST %s < %t

////////////////////////////////////////////////////////////////////////////////
// Class definitions (check for indirect target metadata)

class Cls1 {
public:
  // FT-LABEL: define {{.*}} ptr @_ZN4Cls18receiverEPcPf(
  // FT-SAME: {{.*}} !type [[F_TCLS1RECEIVER:![0-9]+]]
  static int *receiver(char *a, float *b) { return 0; }
};

class Cls2 {
public:
  int *(*fp)(char *, float *);

  // FT-LABEL: define {{.*}} i32 @_ZN4Cls22f1Ecfd(
  // FT-SAME: {{.*}} !type [[F_TCLS2F1:![0-9]+]]
  int f1(char a, float b, double c) { return 0; }

  // FT-LABEL: define {{.*}} ptr @_ZN4Cls22f2EPcPfPd(
  // FT-SAME: {{.*}} !type [[F_TCLS2F2:![0-9]+]]
  int *f2(char *a, float *b, double *c) { return 0; }

  // FT-LABEL: define {{.*}} void @_ZN4Cls22f3E4Cls1(
  // FT-SAME: {{.*}} !type [[F_TCLS2F3F4:![0-9]+]]
  void f3(Cls1 a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls22f4E4Cls1(
  // FT-SAME: {{.*}} !type [[F_TCLS2F3F4]]
  void f4(const Cls1 a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls22f5EP4Cls1(
  // FT-SAME: {{.*}} !type [[F_TCLS2F5:![0-9]+]]
  void f5(Cls1 *a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls22f6EPK4Cls1(
  // FT-SAME: {{.*}} !type [[F_TCLS2F6:![0-9]+]]
  void f6(const Cls1 *a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls22f7ER4Cls1(
  // FT-SAME: {{.*}} !type [[F_TCLS2F7:![0-9]+]]
  void f7(Cls1 &a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls22f8ERK4Cls1(
  // FT-SAME: {{.*}} !type [[F_TCLS2F8:![0-9]+]]
  void f8(const Cls1 &a) {}

  // FT-LABEL: define {{.*}} void @_ZNK4Cls22f9Ev(
  // FT-SAME: {{.*}} !type [[F_TCLS2F9:![0-9]+]]
  void f9() const {}
};

// FT: [[F_TCLS1RECEIVER]] = !{i64 0, !"_ZTSFPiPcPfE.generalized"}
// FT: [[F_TCLS2F1]]   = !{i64 0, !"_ZTSFicfdE.generalized"}
// FT: [[F_TCLS2F2]]   = !{i64 0, !"_ZTSFPiPcPfPdE.generalized"}
// FT: [[F_TCLS2F3F4]] = !{i64 0, !"_ZTSFv4Cls1E.generalized"}
// FT: [[F_TCLS2F5]]   = !{i64 0, !"_ZTSFvP4Cls1E.generalized"}
// FT: [[F_TCLS2F6]]   = !{i64 0, !"_ZTSFvPK4Cls1E.generalized"}
// FT: [[F_TCLS2F7]]   = !{i64 0, !"_ZTSFvR4Cls1E.generalized"}
// FT: [[F_TCLS2F8]]   = !{i64 0, !"_ZTSFvRK4Cls1E.generalized"}
// FT: [[F_TCLS2F9]]   = !{i64 0, !"_ZTSKFvvE.generalized"}

////////////////////////////////////////////////////////////////////////////////
// Callsites (check for indirect callsites' callee_type metadata )

// CST-LABEL: define {{.*}} @_Z3foov
void foo() {
  Cls2 ObjCls2;
  ObjCls2.fp = &Cls1::receiver;

  // CST: call noundef ptr %{{.*}}, !callee_type [[F_TCLS1RECEIVER_CT:![0-9]+]]
  ObjCls2.fp(0, 0);

  auto fp_f1 = &Cls2::f1;
  auto fp_f2 = &Cls2::f2;
  auto fp_f3 = &Cls2::f3;
  auto fp_f4 = &Cls2::f4;
  auto fp_f5 = &Cls2::f5;
  auto fp_f6 = &Cls2::f6;
  auto fp_f7 = &Cls2::f7;
  auto fp_f8 = &Cls2::f8;
  auto fp_f9 = &Cls2::f9;

  Cls2 *ObjCls2Ptr = &ObjCls2;
  Cls1 Cls1Param;

  // CST: call noundef i32 %{{.*}}, !callee_type [[F_TCLS2F1_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f1)(0, 0, 0);

  // CST: call noundef ptr %{{.*}}, !callee_type [[F_TCLS2F2_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f2)(0, 0, 0);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F3F4_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f3)(Cls1Param);

  // CST: call void  %{{.*}}, !callee_type [[F_TCLS2F3F4_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f4)(Cls1Param);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F5_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f5)(&Cls1Param);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F6_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f6)(&Cls1Param);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F7_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f7)(Cls1Param);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F8_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f8)(Cls1Param);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F9_CT:![0-9]+]]
  (ObjCls2Ptr->*fp_f9)();
}

// CST: [[F_TCLS1RECEIVER_CT]] = !{[[F_TCLS1RECEIVER:![0-9]+]]}
// CST: [[F_TCLS1RECEIVER]] = !{i64 0, !"_ZTSFPiPcPfE.generalized"}

// CST: [[F_TCLS2F1_CT]] = !{[[F_TCLS2F1:![0-9]+]]}
// CST: [[F_TCLS2F1]]   = !{i64 0, !"_ZTSFicfdE.generalized"}

// CST: [[F_TCLS2F2_CT]] = !{[[F_TCLS2F2:![0-9]+]]}
// CST: [[F_TCLS2F2]]   = !{i64 0, !"_ZTSFPiPcPfPdE.generalized"}

// CST: [[F_TCLS2F3F4_CT]] = !{[[F_TCLS2F3F4:![0-9]+]]}
// CST: [[F_TCLS2F3F4]] = !{i64 0, !"_ZTSFv4Cls1E.generalized"}

// CST: [[F_TCLS2F5_CT]] = !{[[F_TCLS2F5:![0-9]+]]}
// CST: [[F_TCLS2F5]]   = !{i64 0, !"_ZTSFvP4Cls1E.generalized"}

// CST: [[F_TCLS2F6_CT]] = !{[[F_TCLS2F6:![0-9]+]]}
// CST: [[F_TCLS2F6]]   = !{i64 0, !"_ZTSFvPK4Cls1E.generalized"}

// CST: [[F_TCLS2F7_CT]] = !{[[F_TCLS2F7:![0-9]+]]}
// CST: [[F_TCLS2F7]]   = !{i64 0, !"_ZTSFvR4Cls1E.generalized"}

// CST: [[F_TCLS2F8_CT]] = !{[[F_TCLS2F8:![0-9]+]]}
// CST: [[F_TCLS2F8]]   = !{i64 0, !"_ZTSFvRK4Cls1E.generalized"}

// CST: [[F_TCLS2F9_CT]] = !{[[F_TCLS2F9:![0-9]+]]}
// CST: [[F_TCLS2F9]]   = !{i64 0, !"_ZTSKFvvE.generalized"}
