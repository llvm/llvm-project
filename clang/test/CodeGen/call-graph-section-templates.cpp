// Tests that we assign appropriate identifiers to indirect calls and targets
// specifically for C++ templates.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix=FT    %s < %t
// RUN: FileCheck --check-prefix=CST   %s < %t

////////////////////////////////////////////////////////////////////////////////
// Class definitions and template classes (check for indirect target metadata)

class Cls1 {};

// Cls2 is instantiated with T=Cls1 in foo(). Following checks are for this
// instantiation.
template <class T>
class Cls2 {
public:
  // FT-LABEL: define {{.*}} void @_ZN4Cls2I4Cls1E2f1Ev(
  // FT-SAME: {{.*}} !type [[F_TCLS2F1:![0-9]+]]
  void f1() {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls2I4Cls1E2f2ES0_(
  // FT-SAME: {{.*}} !type [[F_TCLS2F2:![0-9]+]]
  void f2(T a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls2I4Cls1E2f3EPS0_(
  // FT-SAME: {{.*}} !type [[F_TCLS2F3:![0-9]+]]
  void f3(T *a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls2I4Cls1E2f4EPKS0_(
  // FT-SAME: {{.*}} !type [[F_TCLS2F4:![0-9]+]]
  void f4(const T *a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls2I4Cls1E2f5ERS0_(
  // FT-SAME: {{.*}} !type [[F_TCLS2F5:![0-9]+]]
  void f5(T &a) {}

  // FT-LABEL: define {{.*}} void @_ZN4Cls2I4Cls1E2f6ERKS0_(
  // FT-SAME: {{.*}} !type [[F_TCLS2F6:![0-9]+]]
  void f6(const T &a) {}

  // Mixed type function pointer member
  T *(*fp)(T a, T *b, const T *c, T &d, const T &e);
};

// FT: [[F_TCLS2F1]] = !{i64 0, !"_ZTSFvvE.generalized"}
// FT: [[F_TCLS2F2]] = !{i64 0, !"_ZTSFv4Cls1E.generalized"}
// FT: [[F_TCLS2F3]] = !{i64 0, !"_ZTSFvP4Cls1E.generalized"}
// FT: [[F_TCLS2F4]] = !{i64 0, !"_ZTSFvPK4Cls1E.generalized"}
// FT: [[F_TCLS2F5]] = !{i64 0, !"_ZTSFvR4Cls1E.generalized"}
// FT: [[F_TCLS2F6]] = !{i64 0, !"_ZTSFvRK4Cls1E.generalized"}

////////////////////////////////////////////////////////////////////////////////
// Callsites (check for indirect callsite operand bundles)

template <class T>
T *T_func(T a, T *b, const T *c, T &d, const T &e) { return b; }

// CST-LABEL: define {{.*}} @_Z3foov
// CST-SAME: {{.*}} !type [[F_TCLS2F1:![0-9]+]]
void foo() {
  // Methods for Cls2<Cls1> is checked above within the template description.
  Cls2<Cls1> Obj;

  Obj.fp = T_func<Cls1>;
  Cls1 Cls1Obj;
  
  // CST: call noundef ptr %{{.*}}, !callee_type [[F_TFUNC_CLS1_CT:![0-9]+]]
  Obj.fp(Cls1Obj, &Cls1Obj, &Cls1Obj, Cls1Obj, Cls1Obj);

  // Make indirect calls to Cls2's member methods
  auto fp_f1 = &Cls2<Cls1>::f1;
  auto fp_f2 = &Cls2<Cls1>::f2;
  auto fp_f3 = &Cls2<Cls1>::f3;
  auto fp_f4 = &Cls2<Cls1>::f4;
  auto fp_f5 = &Cls2<Cls1>::f5;
  auto fp_f6 = &Cls2<Cls1>::f6;

  auto *Obj2Ptr = &Obj;

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F1_CT:![0-9]+]]
  (Obj2Ptr->*fp_f1)();

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F2_CT:![0-9]+]]
  (Obj2Ptr->*fp_f2)(Cls1Obj);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F3_CT:![0-9]+]]
  (Obj2Ptr->*fp_f3)(&Cls1Obj);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F4_CT:![0-9]+]]
  (Obj2Ptr->*fp_f4)(&Cls1Obj);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F5_CT:![0-9]+]]
  (Obj2Ptr->*fp_f5)(Cls1Obj);

  // CST: call void %{{.*}}, !callee_type [[F_TCLS2F6_CT:![0-9]+]]
  (Obj2Ptr->*fp_f6)(Cls1Obj);
}

// CST-LABEL: define {{.*}} @_Z6T_funcI4Cls1EPT_S1_S2_PKS1_RS1_RS3_(
// CST-SAME: {{.*}} !type [[F_TFUNC_CLS1:![0-9]+]]

// CST: [[F_TCLS2F1]] = !{i64 0, !"_ZTSFvvE.generalized"}
// CST: [[F_TFUNC_CLS1_CT]] = !{[[F_TFUNC_CLS1:![0-9]+]]}
// CST: [[F_TFUNC_CLS1]] = !{i64 0, !"_ZTSFP4Cls1S_S0_PKS_RS_RS1_E.generalized"}
// CST: [[F_TCLS2F1_CT]] = !{[[F_TCLS2F1:![0-9]+]]}
// CST: [[F_TCLS2F2_CT]] = !{[[F_TCLS2F2:![0-9]+]]}
// CST: [[F_TCLS2F2]] = !{i64 0, !"_ZTSFv4Cls1E.generalized"}
// CST: [[F_TCLS2F3_CT]] = !{[[F_TCLS2F3:![0-9]+]]}
// CST: [[F_TCLS2F3]] = !{i64 0, !"_ZTSFvP4Cls1E.generalized"}
// CST: [[F_TCLS2F4_CT]] = !{[[F_TCLS2F4:![0-9]+]]}
// CST: [[F_TCLS2F4]] = !{i64 0, !"_ZTSFvPK4Cls1E.generalized"}
// CST: [[F_TCLS2F5_CT]] = !{[[F_TCLS2F5:![0-9]+]]}
// CST: [[F_TCLS2F5]] = !{i64 0, !"_ZTSFvR4Cls1E.generalized"}
// CST: [[F_TCLS2F6_CT]] = !{[[F_TCLS2F6:![0-9]+]]}
// CST: [[F_TCLS2F6]] = !{i64 0, !"_ZTSFvRK4Cls1E.generalized"}
