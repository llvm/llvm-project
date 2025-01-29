// RUN: %clang_cc1 -triple i686-linux-gnu %s -emit-llvm -o - | FileCheck %s

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

A p1;
C p2;
D p3;

// We end up using an opaque type for 'append' to avoid circular references.
// CHECK: %class.A = type { ptr }
// CHECK: %class.C = type <{ ptr, [4 x i8] }>
// CHECK: %class.D = type { %class.C.base, [3 x i8] }
// CHECK: %class.C.base = type <{ ptr, i8 }>
