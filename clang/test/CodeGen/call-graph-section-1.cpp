// Tests that we assign appropriate identifiers to indirect calls and targets
// specifically for C++ class and instance methods.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fcall-graph-section -S \
// RUN: -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=FT %s < %t
// RUN: FileCheck --check-prefix=CST %s < %t

////////////////////////////////////////////////////////////////////////////////
// Class definitions (check for indirect target metadata)

class Cls1 {
public:
  // FT-DAG: define {{.*}} ptr @_ZN4Cls18receiverEPcPf({{.*}} !type [[F_TCLS1RECEIVER:![0-9]+]]
  static int *receiver(char *a, float *b) { return 0; }
};

class Cls2 {
public:
  int *(*fp)(char *, float *);

  // FT-DAG: define {{.*}} i32 @_ZN4Cls22f1Ecfd({{.*}} !type [[F_TCLS2F1:![0-9]+]]
  int f1(char a, float b, double c) { return 0; }

  // FT-DAG: define {{.*}} ptr @_ZN4Cls22f2EPcPfPd({{.*}} !type [[F_TCLS2F2:![0-9]+]]
  int *f2(char *a, float *b, double *c) { return 0; }

  // FT-DAG: define {{.*}} void @_ZN4Cls22f3E4Cls1({{.*}} !type [[F_TCLS2F3F4:![0-9]+]]
  void f3(Cls1 a) {}

  // FT-DAG: define {{.*}} void @_ZN4Cls22f4E4Cls1({{.*}} !type [[F_TCLS2F3F4]]
  void f4(const Cls1 a) {}

  // FT-DAG: define {{.*}} void @_ZN4Cls22f5EP4Cls1({{.*}} !type [[F_TCLS2F5:![0-9]+]]
  void f5(Cls1 *a) {}

  // FT-DAG: define {{.*}} void @_ZN4Cls22f6EPK4Cls1({{.*}} !type [[F_TCLS2F6:![0-9]+]]
  void f6(const Cls1 *a) {}

  // FT-DAG: define {{.*}} void @_ZN4Cls22f7ER4Cls1({{.*}} !type [[F_TCLS2F7:![0-9]+]]
  void f7(Cls1 &a) {}

  // FT-DAG: define {{.*}} void @_ZN4Cls22f8ERK4Cls1({{.*}} !type [[F_TCLS2F8:![0-9]+]]
  void f8(const Cls1 &a) {}

  // FT-DAG: define {{.*}} void @_ZNK4Cls22f9Ev({{.*}} !type [[F_TCLS2F9:![0-9]+]]
  void f9() const {}
};

// FT-DAG: [[F_TCLS1RECEIVER]] = !{i64 0, !"_ZTSFPvS_S_E.generalized"}
// FT-DAG: [[F_TCLS2F2]]   = !{i64 0, !"_ZTSFPvS_S_S_E.generalized"}
// FT-DAG: [[F_TCLS2F1]]   = !{i64 0, !"_ZTSFicfdE.generalized"}
// FT-DAG: [[F_TCLS2F3F4]] = !{i64 0, !"_ZTSFv4Cls1E.generalized"}
// FT-DAG: [[F_TCLS2F5]]   = !{i64 0, !"_ZTSFvPvE.generalized"}
// FT-DAG: [[F_TCLS2F6]]   = !{i64 0, !"_ZTSFvPKvE.generalized"}
// FT-DAG: [[F_TCLS2F7]]   = !{i64 0, !"_ZTSFvR4Cls1E.generalized"}
// FT-DAG: [[F_TCLS2F8]]   = !{i64 0, !"_ZTSFvRK4Cls1E.generalized"}
// FT-DAG: [[F_TCLS2F9]]   = !{i64 0, !"_ZTSKFvvE.generalized"}

////////////////////////////////////////////////////////////////////////////////
// Callsites (check for indirect callsite operand bundles)

// CST-LABEL: define {{.*}} @_Z3foov
void foo() {
  Cls2 ObjCls2;
  ObjCls2.fp = &Cls1::receiver;

  // CST: call noundef ptr %{{.*}} [ "type"(metadata !"_ZTSFPvS_S_E.generalized") ]
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

  // CST: call noundef i32 %{{.*}} [ "type"(metadata !"_ZTSFicfdE.generalized") ]
  (ObjCls2Ptr->*fp_f1)(0, 0, 0);

  // CST: call noundef ptr %{{.*}} [ "type"(metadata !"_ZTSFPvS_S_S_E.generalized") ]
  (ObjCls2Ptr->*fp_f2)(0, 0, 0);

  // CST: call void %{{.*}} [ "type"(metadata !"_ZTSFv4Cls1E.generalized") ]
  (ObjCls2Ptr->*fp_f3)(Cls1Param);

  // CST: call void  %{{.*}} [ "type"(metadata !"_ZTSFv4Cls1E.generalized") ]
  (ObjCls2Ptr->*fp_f4)(Cls1Param);

  // CST: call void %{{.*}} [ "type"(metadata !"_ZTSFvPvE.generalized") ]
  (ObjCls2Ptr->*fp_f5)(&Cls1Param);

  // CST: call void %{{.*}} [ "type"(metadata !"_ZTSFvPKvE.generalized") ]
  (ObjCls2Ptr->*fp_f6)(&Cls1Param);

  // CST: call void %{{.*}} [ "type"(metadata !"_ZTSFvR4Cls1E.generalized") ]
  (ObjCls2Ptr->*fp_f7)(Cls1Param);

  // CST: call void %{{.*}} [ "type"(metadata !"_ZTSFvRK4Cls1E.generalized") ]
  (ObjCls2Ptr->*fp_f8)(Cls1Param);

  // CST: call void %{{.*}} [ "type"(metadata !"_ZTSKFvvE.generalized") ]
  (ObjCls2Ptr->*fp_f9)();
}
