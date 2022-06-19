// RUN: %clang_cc1 -no-opaque-pointers -triple i686-linux-gnu %s -emit-llvm -o - | FileCheck %s

class A {
  // append has to have the same prototype as fn1 to tickle the bug.
  void (*append)(A *);
};

class B {};
class D;

// C has to be non-C++98 POD with available tail padding, making the LLVM base
// type differ from the complete LLVM type.
class C {
  // This member creates a circular LLVM type reference to %class.D.
  D *m_group;
  B changeListeners;
};
class D : C {};

void fn1(A *p1) {
}

void
fn2(C *) {
}

// We end up using an opaque type for 'append' to avoid circular references.
// CHECK: %class.A = type { {}* }
// CHECK: %class.C = type <{ %class.D*, %class.B, [3 x i8] }>
// CHECK: %class.D = type { %class.C.base, [3 x i8] }
// CHECK: %class.C.base = type <{ %class.D*, %class.B }>
// CHECK: %class.B = type { i8 }
