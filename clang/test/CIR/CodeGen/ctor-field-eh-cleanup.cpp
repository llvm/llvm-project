// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fcxx-exceptions -fexceptions -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Field {
  int x;
  Field(int);
  ~Field();
};

struct TwoFields {
  Field a;
  Field b;
  TwoFields();
};

// When constructing 'b', if its constructor throws, 'a' must be destroyed.
TwoFields::TwoFields() : a(1), b(2) {}

// CIR-LABEL: @_ZN9TwoFieldsC2Ev
// CIR:         cir.call @_ZN5FieldC1Ei
// CIR:         cir.cleanup.scope {
// CIR:           cir.call @_ZN5FieldC1Ei
// CIR:         } cleanup eh {
// CIR:           cir.call @_ZN5FieldD1Ev

// LLVM-LABEL: define dso_local void @_ZN9TwoFieldsC2Ev
// LLVM:         invoke void @_ZN5FieldC1Ei
// LLVM:         landingpad
// LLVM:         call void @_ZN5FieldD1Ev
// LLVM:         resume

// OGCG-LABEL: define dso_local void @_ZN9TwoFieldsC2Ev
// OGCG:         invoke void @_ZN5FieldC1Ei
// OGCG:         landingpad
// OGCG:         call void @_ZN5FieldD1Ev
// OGCG:         resume

struct Base {
  int x;
  Base(int);
  ~Base();
};

struct Derived : Base {
  Field f;
  Derived();
};

// When constructing 'f', if its constructor throws, Base must be destroyed.
Derived::Derived() : Base(1), f(2) {}

// CIR-LABEL: @_ZN7DerivedC2Ev
// CIR:         cir.call @_ZN4BaseC2Ei
// CIR:         cir.cleanup.scope {
// CIR:           cir.call @_ZN5FieldC1Ei
// CIR:         } cleanup eh {
// CIR:           cir.call @_ZN4BaseD2Ev

// LLVM-LABEL: define dso_local void @_ZN7DerivedC2Ev
// LLVM:         invoke void @_ZN5FieldC1Ei
// LLVM:         landingpad
// LLVM:         call void @_ZN4BaseD2Ev
// LLVM:         resume

// OGCG-LABEL: define dso_local void @_ZN7DerivedC2Ev
// OGCG:         invoke void @_ZN5FieldC1Ei
// OGCG:         landingpad
// OGCG:         call void @_ZN4BaseD2Ev
// OGCG:         resume
