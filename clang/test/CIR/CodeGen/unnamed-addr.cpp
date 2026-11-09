// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// Constructors, destructors and virtual member functions are unnamed_addr,
// whether defined, declared, or emitted as an alias of the complete object
// variant. Ordinary member functions are not.

// Aliases are emitted before the function definitions in LLVM IR.
// LLVM: @_ZN1SC1Ev = unnamed_addr alias void (ptr), ptr @_ZN1SC2Ev
// LLVM: @_ZN1SD1Ev = unnamed_addr alias void (ptr), ptr @_ZN1SD2Ev
// LLVM: @_ZN1WD1Ev = unnamed_addr alias void (ptr), ptr @_ZN1WD2Ev

struct S {
  S();
  ~S();
  int x;
};

// The complete object variants (C1/D1) are aliases of the base object
// variants (C2/D2).
S::S() : x(0) {}
// CIR: cir.func {{.*}}dso_local unnamed_addr @_ZN1SC2Ev(
// CIR: cir.func {{.*}}private dso_local unnamed_addr @_ZN1SC1Ev({{.*}}) alias(@_ZN1SC2Ev)
// LLVM: define dso_local void @_ZN1SC2Ev({{.*}}) unnamed_addr

S::~S() {}
// CIR: cir.func {{.*}}dso_local unnamed_addr @_ZN1SD2Ev(
// CIR: cir.func {{.*}}private dso_local unnamed_addr @_ZN1SD1Ev({{.*}}) alias(@_ZN1SD2Ev)
// LLVM: define dso_local void @_ZN1SD2Ev({{.*}}) unnamed_addr

// A virtual member function is unnamed_addr, a non-virtual one is not.
struct V {
  virtual void f();
  void g();
};

void V::f() {}
// CIR: cir.func {{.*}}dso_local unnamed_addr @_ZN1V1fEv(
// LLVM: define dso_local void @_ZN1V1fEv({{.*}}) unnamed_addr

void V::g() {}
// CIR: cir.func {{.*}}dso_local @_ZN1V1gEv(
// LLVM: define dso_local void @_ZN1V1gEv({{.*}}) #

// Declarations of constructors and destructors.
struct D {
  D();
  ~D();
};

void use() { D d; }
// CIR: cir.func private unnamed_addr @_ZN1DC1Ev(
// CIR: cir.func private unnamed_addr @_ZN1DD1Ev(
// LLVM: declare void @_ZN1DC1Ev({{.*}}) unnamed_addr
// LLVM: declare void @_ZN1DD1Ev({{.*}}) unnamed_addr

// A virtual destructor also has a deleting variant (D0), which is its own
// definition rather than an alias.
struct W {
  virtual ~W();
};

W::~W() {}
// CIR: cir.func {{.*}}dso_local unnamed_addr @_ZN1WD2Ev(
// CIR: cir.func {{.*}}private dso_local unnamed_addr @_ZN1WD1Ev({{.*}}) alias(@_ZN1WD2Ev)
// CIR: cir.func {{.*}}dso_local unnamed_addr @_ZN1WD0Ev(
// LLVM: define dso_local void @_ZN1WD2Ev({{.*}}) unnamed_addr
// LLVM: define dso_local void @_ZN1WD0Ev({{.*}}) unnamed_addr

// With a virtual base, the complete object constructor (C1) is a separate
// definition rather than an alias of the base object constructor (C2).
struct VB {
  int b;
};

struct VD : virtual VB {
  VD();
};

VD::VD() {}
// CIR: cir.func {{.*}}dso_local unnamed_addr @_ZN2VDC2Ev(
// CIR: cir.func {{.*}}dso_local unnamed_addr @_ZN2VDC1Ev(
// LLVM: define dso_local void @_ZN2VDC2Ev({{.*}}) unnamed_addr
// LLVM: define dso_local void @_ZN2VDC1Ev({{.*}}) unnamed_addr

// An inline constructor has linkonce_odr linkage and a comdat.
struct I {
  I() {}
};

void use_i() { I i; }
// CIR: cir.func {{.*}}comdat {{.*}}linkonce_odr unnamed_addr @_ZN1IC2Ev(
// LLVM: define linkonce_odr void @_ZN1IC2Ev({{.*}}) unnamed_addr {{.*}}comdat

// A declared virtual member function called non-virtually.
struct V2 {
  virtual void h();
};

void call_decl(V2 &v) { v.V2::h(); }
// CIR: cir.func private unnamed_addr @_ZN2V21hEv(
// LLVM: declare void @_ZN2V21hEv({{.*}}) unnamed_addr
