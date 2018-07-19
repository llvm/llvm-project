// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o %t.ll -fdump-vtable-layouts >%t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

namespace test1 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  // Add an extra virtual method so it's easier to check for the absence of thunks.
  virtual void h();
};

struct X : A, B {
  // CHECK-LABEL: VFTable for 'test1::A' in 'test1::X' (1 entry)
  // CHECK-NEXT:   0 | void test1::X::f()

  // CHECK-LABEL: VFTable for 'test1::B' in 'test1::X' (2 entries)
  // CHECK-NEXT:   0 | void test1::B::g()
  // CHECK-NEXT:   1 | void test1::B::h()

  // CHECK-LABEL: VFTable indices for 'test1::X' (1 entry)
  // CHECK-NEXT:   0 | void test1::X::f()

  // MANGLING-DAG: @"\01??_7X@test1@@6BA@1@@"
  // MANGLING-DAG: @"\01??_7X@test1@@6BB@1@@"

  // Overrides only the left child's method (A::f), needs no thunks.
  virtual void f();
} x;

void build_vftable(X *obj) { obj->f(); }
}

namespace test2 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct X : A, B {
  // CHECK-LABEL: VFTable for 'test2::A' in 'test2::X' (1 entry)
  // CHECK-NEXT:   0 | void test2::A::f()

  // CHECK-LABEL: VFTable for 'test2::B' in 'test2::X' (2 entries)
  // CHECK-NEXT:   0 | void test2::X::g()
  // CHECK-NEXT:   1 | void test2::B::h()

  // CHECK-LABEL: VFTable indices for 'test2::X' (1 entry).
  // CHECK-NEXT:   via vfptr at offset 4
  // CHECK-NEXT:   0 | void test2::X::g()

  // Overrides only the right child's method (B::g), needs this adjustment but
  // not thunks.
  virtual void g();
};

void build_vftable(X *obj) { obj->g(); }
}

namespace test3 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct X : A, B {
  // CHECK-LABEL: VFTable for 'test3::A' in 'test3::X' (2 entries)
  // CHECK-NEXT:   0 | void test3::A::f()
  // CHECK-NEXT:   1 | void test3::X::i()

  // CHECK-LABEL: VFTable for 'test3::B' in 'test3::X' (2 entries)
  // CHECK-NEXT:   0 | void test3::B::g()
  // CHECK-NEXT:   1 | void test3::B::h()

  // CHECK-LABEL: VFTable indices for 'test3::X' (1 entry).
  // CHECK-NEXT:   1 | void test3::X::i()

  // Only adds a new method.
  virtual void i();
};

void build_vftable(X *obj) { obj->i(); }
}

namespace test4 {
struct A {
  virtual void f();
};

struct Empty { };  // Doesn't have a vftable!

// Only the right base has a vftable, so it's laid out before the left one!
struct X : Empty, A {
  // CHECK-LABEL: VFTable for 'test4::A' in 'test4::X' (1 entry)
  // CHECK-NEXT:   0 | void test4::X::f()

  // CHECK-LABEL: VFTable indices for 'test4::X' (1 entry).
  // CHECK-NEXT:   0 | void test4::X::f()

  // MANGLING-DAG: @"\01??_7X@test4@@6B@"

  virtual void f();
} x;

void build_vftable(X *obj) { obj->f(); }
}

namespace test5 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct C : A, B {
  virtual void f();
};

struct X : C {
  // CHECK-LABEL: VFTable for 'test5::A' in 'test5::C' in 'test5::X' (1 entry).
  // CHECK-NEXT:   0 | void test5::X::f()

  // CHECK-LABEL: VFTable for 'test5::B' in 'test5::C' in 'test5::X' (2 entries).
  // CHECK-NEXT:   0 | void test5::B::g()
  // CHECK-NEXT:   1 | void test5::B::h()

  // CHECK-LABEL: VFTable indices for 'test5::X' (1 entry).
  // CHECK-NEXT:   0 | void test5::X::f()

  // MANGLING-DAG: @"\01??_7X@test5@@6BA@1@@"
  // MANGLING-DAG: @"\01??_7X@test5@@6BB@1@@"

  // Overrides both C::f and A::f.
  virtual void f();
} x;

void build_vftable(X *obj) { obj->f(); }
}

namespace test6 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct C : A, B {
  virtual void g();
};

struct X : C {
  // CHECK-LABEL: VFTable for 'test6::A' in 'test6::C' in 'test6::X' (1 entry).
  // CHECK-NEXT:   0 | void test6::A::f()

  // CHECK-LABEL: VFTable for 'test6::B' in 'test6::C' in 'test6::X' (2 entries).
  // CHECK-NEXT:   0 | void test6::X::g()
  // CHECK-NEXT:   1 | void test6::B::h()

  // CHECK-LABEL: VFTable indices for 'test6::X' (1 entry).
  // CHECK-NEXT:   via vfptr at offset 4
  // CHECK-NEXT:   0 | void test6::X::g()

  // Overrides both C::g and B::g.
  virtual void g();
};

void build_vftable(X *obj) { obj->g(); }
}

namespace test7 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct C : A, B {
  // Only adds a new method.
  virtual void i();
};

struct X : C {
  // CHECK-LABEL: VFTable for 'test7::A' in 'test7::C' in 'test7::X' (2 entries).
  // CHECK-NEXT:   0 | void test7::A::f()
  // CHECK-NEXT:   1 | void test7::C::i()

  // CHECK-LABEL: VFTable for 'test7::B' in 'test7::C' in 'test7::X' (2 entries).
  // CHECK-NEXT:   0 | void test7::X::g()
  // CHECK-NEXT:   1 | void test7::B::h()

  // CHECK-LABEL: VFTable indices for 'test7::X' (1 entry).
  // CHECK-NEXT:   via vfptr at offset 4
  // CHECK-NEXT:   0 | void test7::X::g()

  // Overrides grandparent's B::g.
  virtual void g();
};

void build_vftable(X *obj) { obj->g(); }
}

namespace test8 {
struct A {
  virtual void f();
};

struct B : A {
  virtual void g();
};

// There are two 'A' subobjects in this class.
struct X : A, B {
  // CHECK-LABEL: VFTable for 'test8::A' in 'test8::X' (2 entries).
  // CHECK-NEXT:   0 | void test8::A::f()
  // CHECK-NEXT:   1 | void test8::X::h()

  // CHECK-LABEL: VFTable for 'test8::A' in 'test8::B' in 'test8::X' (2 entries).
  // CHECK-NEXT:   0 | void test8::A::f()
  // CHECK-NEXT:   1 | void test8::B::g()

  // CHECK-LABEL: VFTable indices for 'test8::X' (1 entry).
  // CHECK-NEXT:   1 | void test8::X::h()

  // MANGLING-DAG: @"\01??_7X@test8@@6BA@1@@"
  // MANGLING-DAG: @"\01??_7X@test8@@6BB@1@@"

  virtual void h();
} x;

void build_vftable(X *obj) { obj->h(); }
}

namespace test9 {
struct A {
  virtual void f();
};

struct B {
  virtual void g();
  virtual void h();
};

struct C : A, B {
  // Overrides only the left child's method (A::f).
  virtual void f();
};

struct D : A, B {
  // Overrides only the right child's method (B::g).
  virtual void g();
};

// 2-level structure with repeating subobject types, but no thunks needed.
struct X : C, D {
  // CHECK-LABEL: VFTable for 'test9::A' in 'test9::C' in 'test9::X' (2 entries)
  // CHECK-NEXT:   0 | void test9::C::f()
  // CHECK-NEXT:   1 | void test9::X::z()

  // CHECK-LABEL: VFTable for 'test9::B' in 'test9::C' in 'test9::X' (2 entries)
  // CHECK-NEXT:   0 | void test9::B::g()
  // CHECK-NEXT:   1 | void test9::B::h()

  // CHECK-LABEL: VFTable for 'test9::A' in 'test9::D' in 'test9::X' (1 entry)
  // CHECK-NEXT:   0 | void test9::A::f()

  // CHECK-LABEL: VFTable for 'test9::B' in 'test9::D' in 'test9::X' (2 entries)
  // CHECK-NEXT:   0 | void test9::D::g()
  // CHECK-NEXT:   1 | void test9::B::h()

  // CHECK-LABEL: VFTable indices for 'test9::X' (1 entry).
  // CHECK-NEXT:   1 | void test9::X::z()

  // MANGLING-DAG: @"\01??_7X@test9@@6BA@1@C@1@@"
  // MANGLING-DAG: @"\01??_7X@test9@@6BA@1@D@1@@"
  // MANGLING-DAG: @"\01??_7X@test9@@6BB@1@C@1@@"
  // MANGLING-DAG: @"\01??_7X@test9@@6BB@1@D@1@@"

  virtual void z();
} x;

void build_vftable(test9::X *obj) { obj->z(); }
}
