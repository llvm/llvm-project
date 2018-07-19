// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o %t.ll -fdump-vtable-layouts >%t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

struct A {
  virtual void f();
};

struct B {
  virtual void g() = 0;
  virtual void h();
};

struct C : A, B {
  // CHECK-LABEL: VFTable for 'A' in 'C' (1 entry)
  // CHECK-NEXT:   0 | void A::f()

  // CHECK-LABEL: VFTable for 'B' in 'C' (2 entries)
  // CHECK-NEXT:   0 | void C::g()
  // CHECK-NEXT:   1 | void B::h()

  // CHECK-LABEL: VFTable indices for 'C' (1 entry).
  // CHECK-NEXT:   via vfptr at offset 4
  // CHECK-NEXT:   0 | void C::g()

  // MANGLING-DAG: @"\01??_7C@@6BA@@@"
  // MANGLING-DAG: @"\01??_7C@@6BB@@@"

  // Overrides only the right child's method (B::g),
  // needs this adjustment but not thunks.
  virtual void g();
};

C c;
void build_vftable(C *obj) { obj->g(); }
