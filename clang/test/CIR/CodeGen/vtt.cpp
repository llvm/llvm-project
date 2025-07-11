// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll  %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll  %s

// Note: This test will be expanded to verify VTT emission and VTT implicit
// argument handling. For now, it's just test the record layout.

class A {
public:
  int a;
  virtual void v() {}
};

class B : public virtual A {
public:
  int b;
  virtual void w();
};

class C : public virtual A {
public:
  long c;
  virtual void x() {}
};

class D : public B, public C {
public:
  long d;
  virtual void y() {}
};

// This is just here to force the record types to be emitted.
void f(D *d) {}

// CIR: !rec_A2Ebase = !cir.record<struct "A.base" packed {!cir.vptr, !s32i}>
// CIR: !rec_B2Ebase = !cir.record<struct "B.base" packed {!cir.vptr, !s32i}>
// CIR: !rec_C2Ebase = !cir.record<struct "C.base" {!cir.vptr, !s64i}>
// CIR: !rec_D = !cir.record<class "D" {!rec_B2Ebase, !rec_C2Ebase, !s64i, !rec_A2Ebase}>

// Nothing interesting to see here yet.
// LLVM: define{{.*}} void @_Z1fP1D
// OGCG: define{{.*}} void @_Z1fP1D
