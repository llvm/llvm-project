// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o %t.ll -fdump-vtable-layouts >%t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

namespace test1 {
struct A {
  virtual void g();
  // Add an extra virtual method so it's easier to check for the absence of thunks.
  virtual void h();
};

struct B {
  virtual void g();
};

// Overrides a method of two bases at the same time, thus needing thunks.
struct C : A, B {
  virtual void g();
};

struct D {
  virtual B* foo();
  virtual void z();
};

struct X : D {
  // CHECK-LABEL: VFTable for 'test1::D' in 'test1::X' (3 entries).
  // CHECK-NEXT:   0 | test1::C *test1::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test1::B *'): 4 non-virtual]
  // CHECK-NEXT:   1 | void test1::D::z()
  // CHECK-NEXT:   2 | test1::C *test1::X::foo()

  // CHECK-LABEL: Thunks for 'test1::C *test1::X::foo()' (1 entry).
  // CHECK-NEXT:   0 | [return adjustment (to type 'struct test1::B *'): 4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test1::X' (1 entry).
  // CHECK-NEXT:   2 | test1::C *test1::X::foo()

  // MANGLING-DAG: @"??_7X@test1@@6B@"

  virtual C* foo();
} x;

void build_vftable(X *obj) { obj->foo(); }
}

namespace test2 {
struct A {
  virtual void g();
  virtual void h();
};

struct B {
  virtual void g();
};

struct C : A, B {
  virtual void g();
};

struct D {
  virtual B* foo();
  virtual void z();
};

struct E : D {
  virtual C* foo();
};

struct F : C { };

struct X : E {
  virtual F* foo();
  // CHECK-LABEL: VFTable for 'test2::D' in 'test2::E' in 'test2::X' (4 entries).
  // CHECK-NEXT:   0 | test2::F *test2::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test2::B *'): 4 non-virtual]
  // CHECK-NEXT:   1 | void test2::D::z()
  // CHECK-NEXT:   2 | test2::F *test2::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test2::C *'): 0 non-virtual]
  // CHECK-NEXT:   3 | test2::F *test2::X::foo()

  // CHECK-LABEL: Thunks for 'test2::F *test2::X::foo()' (2 entries).
  // CHECK-NEXT:   0 | [return adjustment (to type 'struct test2::C *'): 0 non-virtual]
  // CHECK-NEXT:   1 | [return adjustment (to type 'struct test2::B *'): 4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test2::X' (1 entry).
  // CHECK-NEXT:   3 | test2::F *test2::X::foo()
};

void build_vftable(X *obj) { obj->foo(); }
}

namespace test3 {
struct A {
  virtual void g();
  virtual void h();
};

struct B {
  virtual void g();
};

struct C : A, B {
  virtual void g();
};

struct D {
  virtual B* foo();
  virtual void z();
};

struct E : D {
  virtual C* foo();
};

struct F : A, C { };

struct X : E {
  // CHECK-LABEL: VFTable for 'test3::D' in 'test3::E' in 'test3::X' (4 entries).
  // CHECK-NEXT:   0 | test3::F *test3::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test3::B *'): 8 non-virtual]
  // CHECK-NEXT:   1 | void test3::D::z()
  // CHECK-NEXT:   2 | test3::F *test3::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test3::C *'): 4 non-virtual]
  // CHECK-NEXT:   3 | test3::F *test3::X::foo()

  // CHECK-LABEL: Thunks for 'test3::F *test3::X::foo()' (2 entries).
  // CHECK-NEXT:   0 | [return adjustment (to type 'struct test3::C *'): 4 non-virtual]
  // CHECK-NEXT:   1 | [return adjustment (to type 'struct test3::B *'): 8 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test3::X' (1 entry).
  // CHECK-NEXT:   3 | test3::F *test3::X::foo()

  virtual F* foo();
};

void build_vftable(X *obj) { obj->foo(); }
}

namespace test4 {
struct A {
  virtual void g();
  virtual void h();
};

struct B {
  virtual void g();
};

struct C : A, B {
  virtual void g();
};

struct D {
  virtual B* foo();
  virtual void z();
};

struct E : D {
  virtual C* foo();
};

struct F : A, C { };

struct X : D, E {
  // CHECK-LABEL: VFTable for 'test4::D' in 'test4::X' (3 entries).
  // CHECK-NEXT:   0 | test4::F *test4::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test4::B *'): 8 non-virtual]
  // CHECK-NEXT:   1 | void test4::D::z()
  // CHECK-NEXT:   2 | test4::F *test4::X::foo()

  // CHECK-LABEL: Thunks for 'test4::F *test4::X::foo()' (1 entry).
  // CHECK-NEXT:   0 | [return adjustment (to type 'struct test4::B *'): 8 non-virtual]

  // CHECK-LABEL: VFTable for 'test4::D' in 'test4::E' in 'test4::X' (4 entries).
  // CHECK-NEXT:   0 | test4::F *test4::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test4::B *'): 8 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   1 | void test4::D::z()
  // CHECK-NEXT:   2 | test4::F *test4::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test4::C *'): 4 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   3 | test4::F *test4::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test4::F *'): 0 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'test4::F *test4::X::foo()' (3 entries).
  // CHECK-NEXT:   0 | [return adjustment (to type 'struct test4::F *'): 0 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   1 | [return adjustment (to type 'struct test4::C *'): 4 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   2 | [return adjustment (to type 'struct test4::B *'): 8 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test4::X' (1 entry).
  // CHECK-NEXT:   2 | test4::F *test4::X::foo()

  virtual F* foo();
};

void build_vftable(X *obj) { obj->foo(); }
}

namespace test5 {
struct A {
  virtual void g();
  virtual void h();
};

struct B {
  virtual void g();
};

struct C : A, B {
  virtual void g();
};

struct D {
  virtual B* foo();
  virtual void z();
};

struct X : A, D {
  // CHECK-LABEL: VFTable for 'test5::A' in 'test5::X' (2 entries).
  // CHECK-NEXT:   0 | void test5::A::g()
  // CHECK-NEXT:   1 | void test5::A::h()

  // CHECK-LABEL: VFTable for 'test5::D' in 'test5::X' (3 entries).
  // CHECK-NEXT:   0 | test5::C *test5::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test5::B *'): 4 non-virtual]
  // CHECK-NEXT:   1 | void test5::D::z()
  // CHECK-NEXT:   2 | test5::C *test5::X::foo()

  // CHECK-LABEL: Thunks for 'test5::C *test5::X::foo()' (1 entry).
  // CHECK-NEXT:   0 | [return adjustment (to type 'struct test5::B *'): 4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test5::X' (1 entry).
  // CHECK-NEXT:   via vfptr at offset 4
  // CHECK-NEXT:   2 | test5::C *test5::X::foo()

  virtual C* foo();
};

void build_vftable(X *obj) { obj->foo(); }
}

namespace test6 {
struct A {
  virtual void g();
  virtual void h();
};

struct B {
  virtual void g();
};

struct C : A, B {
  virtual void g();
};

struct D {
  virtual B* foo();
  virtual void z();
};

struct E : A, D {
  virtual C* foo();
};

struct F : A, C { };

struct X : E {
  // CHECK-LABEL: VFTable for 'test6::A' in 'test6::E' in 'test6::X' (2 entries).
  // CHECK-NEXT:   0 | void test6::A::g()
  // CHECK-NEXT:   1 | void test6::A::h()

  // CHECK-LABEL: VFTable for 'test6::D' in 'test6::E' in 'test6::X' (4 entries).
  // CHECK-NEXT:   0 | test6::F *test6::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test6::B *'): 8 non-virtual]
  // CHECK-NEXT:   1 | void test6::D::z()
  // CHECK-NEXT:   2 | test6::F *test6::X::foo()
  // CHECK-NEXT:       [return adjustment (to type 'struct test6::C *'): 4 non-virtual]
  // CHECK-NEXT:   3 | test6::F *test6::X::foo()

  // CHECK-LABEL: Thunks for 'test6::F *test6::X::foo()' (2 entries).
  // CHECK-NEXT:   0 | [return adjustment (to type 'struct test6::C *'): 4 non-virtual]
  // CHECK-NEXT:   1 | [return adjustment (to type 'struct test6::B *'): 8 non-virtual]

  // CHECK-LABEL: VFTable indices for 'test6::X' (1 entry).
  // CHECK-NEXT:   -- accessible via vfptr at offset 4 --
  // CHECK-NEXT:   3 | test6::F *test6::X::foo()

  virtual F* foo();
};

void build_vftable(X *obj) { obj->foo(); }
}

namespace test7 {
struct A {
  virtual A *f() = 0;
};
struct B {
  virtual void g();
};
struct C : B, A {
  virtual void g();
  virtual C *f() = 0;
  // CHECK-LABEL: VFTable for 'test7::B' in 'test7::C' (1 entry).
  // CHECK-NEXT:   0 | void test7::C::g()

  // CHECK-LABEL: VFTable for 'test7::A' in 'test7::C' (2 entries).
  // CHECK-NEXT:   0 | test7::C *test7::C::f() [pure]
  // CHECK-NEXT:   1 | test7::C *test7::C::f() [pure]

  // No return adjusting thunks needed for pure virtual methods.
  // CHECK-NOT: Thunks for 'test7::C *test7::C::f()'
};

void build_vftable(C *obj) { obj->g(); }
}

namespace pr20444 {
struct A {
  virtual A* f();
};
struct B {
  virtual B* f();
};
struct C : A, B {
  virtual C* f();
  // CHECK-LABEL: VFTable for 'pr20444::A' in 'pr20444::C' (1 entry).
  // CHECK-NEXT:   0 | pr20444::C *pr20444::C::f()

  // CHECK-LABEL: VFTable for 'pr20444::B' in 'pr20444::C' (2 entries).
  // CHECK-NEXT:   0 | pr20444::C *pr20444::C::f()
  // CHECK-NEXT:       [return adjustment (to type 'struct pr20444::B *'): 4 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   1 | pr20444::C *pr20444::C::f()
  // CHECK-NEXT:       [return adjustment (to type 'struct pr20444::C *'): 0 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
};

void build_vftable(C *obj) { obj->f(); }

struct D : C {
  virtual D* f();
  // CHECK-LABEL: VFTable for 'pr20444::A' in 'pr20444::C' in 'pr20444::D' (1 entry).
  // CHECK-NEXT:   0 | pr20444::D *pr20444::D::f()

  // CHECK-LABEL: VFTable for 'pr20444::B' in 'pr20444::C' in 'pr20444::D' (3 entries).
  // CHECK-NEXT:   0 | pr20444::D *pr20444::D::f()
  // CHECK-NEXT:       [return adjustment (to type 'struct pr20444::B *'): 4 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   1 | pr20444::D *pr20444::D::f()
  // CHECK-NEXT:       [return adjustment (to type 'struct pr20444::C *'): 0 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   2 | pr20444::D *pr20444::D::f()
  // CHECK-NEXT:       [return adjustment (to type 'struct pr20444::D *'): 0 non-virtual]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
};

void build_vftable(D *obj) { obj->f(); }
}
