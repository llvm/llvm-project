// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// TODO(cir): Try to emit base destructor as an alias at O1 or higher.

// FIXME: LLVM IR dialect does not yet support function ptr globals, which precludes
// a lot of the proper semantics for properly representing alias functions in LLVM
// (see the note on LLVM_O1 below).

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

B::~B() { }

// Aliases are inserted before the function definitions in LLVM IR
// FIXME: These should have unnamed_addr set.
// LLVM: @_ZN1BD1Ev = alias void (ptr), ptr @_ZN1BD2Ev
// LLVM: @_ZN1CD1Ev = alias void (ptr), ptr @_ZN1CD2Ev

// OGCG: @_ZN1BD1Ev = unnamed_addr alias void (ptr), ptr @_ZN1BD2Ev
// OGCG: @_ZN1CD1Ev = unnamed_addr alias void (ptr), ptr @_ZN1CD2Ev


// Base (D2) dtor for B: calls A's base dtor.

// CIR: cir.func{{.*}} @_ZN1BD2Ev
// CIR:   cir.call @_ZN6MemberD1Ev
// CIR:   cir.call @_ZN1AD2Ev

// LLVM: define{{.*}} void @_ZN1BD2Ev
// LLVM:   call void @_ZN6MemberD1Ev
// LLVM:   call void @_ZN1AD2Ev

// OGCG: define{{.*}} @_ZN1BD2Ev
// OGCG:   call void @_ZN6MemberD1Ev
// OGCG:   call void @_ZN1AD2Ev

// Complete (D1) dtor for B: just an alias because there are no virtual bases.

// CIR: cir.func{{.*}} @_ZN1BD1Ev(!cir.ptr<!rec_B>) alias(@_ZN1BD2Ev)
// This is defined above for LLVM and OGCG.

// Deleting (D0) dtor for B: defers to the complete dtor but also calls operator delete.

// CIR: cir.func{{.*}} @_ZN1BD0Ev
// CIR:   cir.call @_ZN1BD1Ev(%[[THIS:.*]]) nothrow : (!cir.ptr<!rec_B>) -> ()
// CIR:   %[[THIS_VOID:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<!rec_B> -> !cir.ptr<!void>
// CIR:   %[[SIZE:.*]] = cir.const #cir.int<16>
// CIR:   cir.call @_ZdlPvm(%[[THIS_VOID]], %[[SIZE]])

// LLVM: define{{.*}} void @_ZN1BD0Ev
// LLVM:   call void @_ZN1BD1Ev(ptr %[[THIS:.*]])
// LLVM:   call void @_ZdlPvm(ptr %[[THIS]], i64 16)

// OGCG: define{{.*}} @_ZN1BD0Ev
// OGCG:   call void @_ZN1BD1Ev(ptr{{.*}} %[[THIS:.*]])
// OGCG:   call void @_ZdlPvm(ptr{{.*}} %[[THIS]], i64{{.*}} 16)

struct C : B {
  ~C();
};

C::~C() { }

// Base (D2) dtor for C: calls B's base dtor.

// CIR: cir.func{{.*}} @_ZN1CD2Ev
// CIR:   %[[B:.*]] = cir.base_class_addr %[[THIS:.*]] : !cir.ptr<!rec_C> nonnull [0] -> !cir.ptr<!rec_B>
// CIR:   cir.call @_ZN1BD2Ev(%[[B]])

// LLVM: define{{.*}} void @_ZN1CD2Ev
// LLVM:   call void @_ZN1BD2Ev

// OGCG: define{{.*}} @_ZN1CD2Ev
// OGCG:   call void @_ZN1BD2Ev

// Complete (D1) dtor for C: just an alias because there are no virtual bases.

// CIR: cir.func{{.*}} @_ZN1CD1Ev(!cir.ptr<!rec_C>) alias(@_ZN1CD2Ev)
// This is defined above for LLVM and OGCG.


// Deleting (D0) dtor for C: defers to the complete dtor but also calls operator delete.

// CIR: cir.func{{.*}} @_ZN1CD0Ev
// CIR:   cir.call @_ZN1CD1Ev(%[[THIS:.*]]) nothrow : (!cir.ptr<!rec_C>) -> ()
// CIR:   %[[THIS_VOID:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<!rec_C> -> !cir.ptr<!void>
// CIR:   %[[SIZE:.*]] = cir.const #cir.int<16>
// CIR:   cir.call @_ZdlPvm(%[[THIS_VOID]], %[[SIZE]])

// LLVM: define{{.*}} void @_ZN1CD0Ev
// LLVM:   call void @_ZN1CD1Ev(ptr %[[THIS:.*]])
// LLVM:   call void @_ZdlPvm(ptr %[[THIS]], i64 16)

// OGCG: define{{.*}} @_ZN1CD0Ev
// OGCG:   call void @_ZN1CD1Ev(ptr{{.*}} %[[THIS:.*]])
// OGCG:   call void @_ZdlPvm(ptr{{.*}} %[[THIS]], i64{{.*}} 16)

namespace PR12798 {
  // A qualified call to a base class destructor should not undergo virtual
  // dispatch. Template instantiation used to lose the qualifier.
  struct A { virtual ~A(); };
  template<typename T> void f(T *p) { p->A::~A(); }

  // CIR: cir.func{{.*}} @_ZN7PR127981fINS_1AEEEvPT_
  // CIR:   cir.call @_ZN7PR127981AD1Ev

  // LLVM: define{{.*}} @_ZN7PR127981fINS_1AEEEvPT_
  // LLVM:   call void @_ZN7PR127981AD1Ev

  // OGCG: define{{.*}} @_ZN7PR127981fINS_1AEEEvPT_
  // OGCG:   call void @_ZN7PR127981AD1Ev

  template void f(A*);
}
