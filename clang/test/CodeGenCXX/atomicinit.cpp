// RUN: %clang_cc1 %s -emit-llvm -O1 -o - -triple=i686-apple-darwin9 | FileCheck %s
struct A {
  _Atomic(int) i;
  A(int j);
  void v(int j);
};
// Storing to atomic values should be atomic
// CHECK: store atomic i32
void A::v(int j) { i = j; }
// Initialising atomic values should not be atomic
// CHECK-NOT: store atomic 
A::A(int j) : i(j) {}

struct B {
  int i;
  B(int x) : i(x) {}
};

_Atomic(B) b;

// CHECK: define void @_Z11atomic_initR1Ai
void atomic_init(A& a, int i) {
  // CHECK-NOT: atomic
  // CHECK: tail call void @_ZN1BC1Ei
  __c11_atomic_init(&b, B(i));
  // CHECK-NEXT: ret void
}

// CHECK: define void @_Z16atomic_init_boolPU7_Atomicbb
void atomic_init_bool(_Atomic(bool) *ab, bool b) {
  // CHECK-NOT: atomic
  // CHECK: {{zext i1.*to i8}}
  // CHECK-NEXT: store i8
  __c11_atomic_init(ab, b);
  // CHECK-NEXT: ret void
}

struct AtomicBoolMember {
  _Atomic(bool) ab;
  AtomicBoolMember(bool b);
};

// CHECK: define void @_ZN16AtomicBoolMemberC2Eb
// CHECK: {{zext i1.*to i8}}
// CHECK-NEXT: store i8
// CHECK-NEXT: ret void
AtomicBoolMember::AtomicBoolMember(bool b) : ab(b) { }

