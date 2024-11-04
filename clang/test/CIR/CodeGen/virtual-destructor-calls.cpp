
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// PREV: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -mconstructor-aliases -O1 -disable-llvm-passes | FileCheck %s

struct Member {
  ~Member();
};

struct A {
  virtual ~A();
};

struct B : A {
  Member m;
  virtual ~B();
};

// Base dtor: actually calls A's base dtor.
// CIR: cir.func @_ZN1BD2Ev
// CIR:   cir.call @_ZN6MemberD1Ev
// CIR:   cir.call @_ZN1AD2Ev
// LLVM: define{{.*}} void @_ZN1BD2Ev(ptr
// LLVM: call void @_ZN6MemberD1Ev
// LLVM: call void @_ZN1AD2Ev

// Complete dtor: just an alias because there are no virtual bases.
// CIR: cir.func private @_ZN1BD1Ev(!cir.ptr<!ty_B>) alias(@_ZN1BD2Ev)
// FIXME: LLVM output should be: @_ZN1BD1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN1BD2Ev
// LLVM: declare {{.*}} dso_local void @_ZN1BD1Ev(ptr)

// Deleting dtor: defers to the complete dtor.
// LLVM: define{{.*}} void @_ZN1BD0Ev(ptr
// LLVM: call void @_ZN1BD1Ev
// LLVM: call void @_ZdlPv

// (aliases from C)
// FIXME: this should be an alias declaration.
// CIR: cir.func @_ZN1CD2Ev(%arg0: !cir.ptr<!ty_C>{{.*}})) {{.*}} {
// CIR: cir.func private @_ZN1CD1Ev(!cir.ptr<!ty_C>) alias(@_ZN1CD2Ev)

// FIXME: LLVM output should be: @_ZN1CD2Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN1BD2Ev
// LLVM: define dso_local void @_ZN1CD2Ev(ptr
// FIXME: LLVM output should be: @_ZN1CD1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN1CD2Ev
// LLVM: declare {{.*}} dso_local void @_ZN1CD1Ev(ptr)

B::~B() { }

struct C : B {
  ~C();
};

C::~C() { }

// Complete dtor: just an alias (checked above).

// Deleting dtor: defers to the complete dtor.
// CIR: cir.func @_ZN1CD0Ev
// CIR: cir.call @_ZN1CD1Ev
// CIR: cir.call @_ZdlPvm
// LLVM: define{{.*}} void @_ZN1CD0Ev(ptr
// LLVM: call void @_ZN1CD1Ev
// LLVM: call void @_ZdlPv

// Base dtor: just an alias to B's base dtor.

namespace PR12798 {
  // A qualified call to a base class destructor should not undergo virtual
  // dispatch. Template instantiation used to lose the qualifier.
  struct A { virtual ~A(); };
  template<typename T> void f(T *p) { p->A::~A(); }

  // CIR: cir.func weak_odr @_ZN7PR127981fINS_1AEEEvPT_
  // CIR: cir.call @_ZN7PR127981AD1Ev
  // LLVM: define {{.*}} @_ZN7PR127981fINS_1AEEEvPT_(
  // LLVM: call void @_ZN7PR127981AD1Ev(
  template void f(A*);
}
