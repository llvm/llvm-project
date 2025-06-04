// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

template<class T>
class X {
public:
  int f() { return 0; }
};

template<> class X<int> {
public:
  int f() { return 1; }
};

// TODO: This will get dropped when we are deferring functions
// The speecialization is instantiated first
// CIR: cir.func{{.*}} @_ZN1XIiE1fEv
// CIR:   cir.const #cir.int<1>

// LLVM: define{{.*}} i32 @_ZN1XIiE1fEv
// LLVM:   store i32 1

void test_double() {
  X<double> d;
  d.f();
}

// CIR: cir.func{{.*}} @_ZN1XIdE1fEv
// CIR:   cir.const #cir.int<0>
//
// CIR: cir.func{{.*}} @_Z11test_doublev()
// CIR:   cir.call @_ZN1XIdE1fEv

// LLVM: define{{.*}} i32 @_ZN1XIdE1fEv
// LLVM:   store i32 0
//
// LLVM: define{{.*}} void @_Z11test_doublev()
// LLVM:   call i32 @_ZN1XIdE1fEv

// OGCG: define{{.*}} void @_Z11test_doublev()
// OGCG:   call{{.*}} i32 @_ZN1XIdE1fEv
//
// OGCG: define{{.*}} i32 @_ZN1XIdE1fEv
// OGCG:   ret i32 0

void test_int() {
  X<int> n;
  n.f();
}

// CIR: cir.func{{.*}} @_Z8test_intv()
// CIR:   cir.call @_ZN1XIiE1fEv

// LLVM: define{{.*}} void @_Z8test_intv()
// LLVM:   call i32 @_ZN1XIiE1fEv

// OGCG: define{{.*}} void @_Z8test_intv()
// OGCG:   call{{.*}} i32 @_ZN1XIiE1fEv
//
// OGCG: define{{.*}} i32 @_ZN1XIiE1fEv
// OGCG:   ret i32 1

void test_short() {
  X<short> s;
  s.f();
}

// CIR: cir.func{{.*}} @_ZN1XIsE1fEv
// CIR:   cir.const #cir.int<0>
//
// CIR: cir.func{{.*}} @_Z10test_shortv()
// CIR:   cir.call @_ZN1XIsE1fEv

// LLVM: define{{.*}} i32 @_ZN1XIsE1fEv
// LLVM: store i32 0
//
// LLVM: define{{.*}} void @_Z10test_shortv()
// LLVM:   call i32 @_ZN1XIsE1fEv

// OGCG: define{{.*}} void @_Z10test_shortv()
// OGCG:   call{{.*}} i32 @_ZN1XIsE1fEv
//
// OGCG: define{{.*}} i32 @_ZN1XIsE1fEv
// OGCG:   ret i32 0
