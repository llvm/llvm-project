// RUN: %clang_cc1 -std=c++11 -fms-extensions -fno-rtti -emit-llvm -o %t.ll -fdump-vtable-layouts %s -triple=i386-pc-win32 >%t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

struct Empty { };

struct A {
  virtual void f();
  virtual void z();  // Useful to check there are no thunks for f() when appropriate.
};

struct B {
  virtual void g();
};

struct C: virtual A {
  // CHECK-LABEL: VFTable for 'A' in 'C' (2 entries)
  // CHECK-NEXT: 0 | void C::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable indices for 'C' (1 entry)
  // CHECK-NEXT: vbtable index 1, vfptr at offset 0
  // CHECK-NEXT: 0 | void C::f()

  // MANGLING-DAG: @"\01??_7C@@6B@"

  virtual void f() {}
};

C c;
void use(C *obj) { obj->f(); }

struct D: virtual A {
  // CHECK-LABEL: VFTable for 'D' (1 entry).
  // CHECK-NEXT: 0 | void D::h()

  // CHECK-LABEL: VFTable for 'A' in 'D' (2 entries).
  // CHECK-NEXT: 0 | void D::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable indices for 'D' (2 entries).
  // CHECK-NEXT: via vfptr at offset 0
  // CHECK-NEXT: 0 | void D::h()
  // CHECK-NEXT: via vbtable index 1, vfptr at offset 0
  // CHECK-NEXT: 0 | void D::f()

  // MANGLING-DAG: @"\01??_7D@@6B0@@"
  // MANGLING-DAG: @"\01??_7D@@6BA@@@"

  virtual void f();
  virtual void h();
};

D d;
void use(D *obj) { obj->h(); }

namespace Test1 {

struct X { int x; };

// X and A get reordered in the layout since X doesn't have a vfptr while A has.
struct Y : X, A { };
// MANGLING-DAG: @"\01??_7Y@Test1@@6B@"

struct Z : virtual Y {
  Z();
  // CHECK-LABEL: VFTable for 'A' in 'Test1::Y' in 'Test1::Z' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-NOT: VFTable indices for 'Test1::Z'

  // MANGLING-DAG: @"\01??_7Z@Test1@@6B@"
};

Z::Z() {}
}

namespace Test2 {

struct X: virtual A, virtual B {
  // CHECK-LABEL: VFTable for 'Test2::X' (1 entry).
  // CHECK-NEXT: 0 | void Test2::X::h()

  // CHECK-LABEL: VFTable for 'A' in 'Test2::X' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable for 'B' in 'Test2::X' (1 entry).
  // CHECK-NEXT: 0 | void B::g()

  // CHECK-LABEL: VFTable indices for 'Test2::X' (1 entry).
  // CHECK-NEXT: 0 | void Test2::X::h()

  // MANGLING-DAG: @"\01??_7X@Test2@@6B01@@"
  // MANGLING-DAG: @"\01??_7X@Test2@@6BA@@@"
  // MANGLING-DAG: @"\01??_7X@Test2@@6BB@@@"

  virtual void h();
};

X x;
void use(X *obj) { obj->h(); }
}

namespace Test3 {

struct X : virtual A {
  // MANGLING-DAG: @"\01??_7X@Test3@@6B@"
};

struct Y: virtual X {
  Y();
  // CHECK-LABEL: VFTable for 'A' in 'Test3::X' in 'Test3::Y' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-NOT: VFTable indices for 'Test3::Y'

  // MANGLING-DAG: @"\01??_7Y@Test3@@6B@"
};

Y::Y() {}
}

namespace Test4 {

struct X: virtual C {
  X();
  // This one's interesting. C::f expects (A*) to be passed as 'this' and does
  // ECX-=4 to cast to (C*). In X, C and A vbases are reordered, so the thunk
  // should pass a pointer to the end of X in order
  // for ECX-=4 to point at the C part.

  // CHECK-LABEL: VFTable for 'A' in 'C' in 'Test4::X' (2 entries).
  // CHECK-NEXT: 0 | void C::f()
  // CHECK-NEXT:     [this adjustment: 8 non-virtual]
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: Thunks for 'void C::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: 8 non-virtual]

  // CHECK-NOT: VFTable indices for 'Test4::X'

  // MANGLING-DAG: @"\01??_7X@Test4@@6B@"

  // Also check the mangling of the thunk.
  // MANGLING-DAG: define linkonce_odr dso_local x86_thiscallcc void @"\01?f@C@@WPPPPPPPI@AEXXZ"
};

X::X() {}
}

namespace Test5 {

// New methods are added to the base's vftable.
struct X : A {
  // MANGLING-DAG: @"\01??_7X@Test5@@6B@"
  virtual void g();
};

struct Y : virtual X {
  // CHECK-LABEL: VFTable for 'Test5::Y' (1 entry).
  // CHECK-NEXT: 0 | void Test5::Y::h()

  // CHECK-LABEL: VFTable for 'A' in 'Test5::X' in 'Test5::Y' (3 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()
  // CHECK-NEXT: 2 | void Test5::X::g()

  // CHECK-LABEL: VFTable indices for 'Test5::Y' (1 entry).
  // CHECK-NEXT: 0 | void Test5::Y::h()

  // MANGLING-DAG: @"\01??_7Y@Test5@@6B01@@"
  // MANGLING-DAG: @"\01??_7Y@Test5@@6BX@1@@"

  virtual void h();
};

Y y;
void use(Y *obj) { obj->h(); }
}

namespace Test6 {

struct X : A, virtual Empty {
  X();
  // CHECK-LABEL: VFTable for 'A' in 'Test6::X' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-NOT: VFTable indices for 'Test6::X'

  // MANGLING-DAG: @"\01??_7X@Test6@@6B@"
};

X::X() {}
}

namespace Test7 {

struct X : C {
  // MANGLING-DAG: @"\01??_7X@Test7@@6B@"
};

struct Y : virtual X {
  Y();
  // CHECK-LABEL: VFTable for 'A' in 'C' in 'Test7::X' in 'Test7::Y' (2 entries).
  // CHECK-NEXT: 0 | void C::f()
  // CHECK-NEXT:     [this adjustment: 8 non-virtual]
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: Thunks for 'void C::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: 8 non-virtual]

  // CHECK-NOT: VFTable indices for 'Test7::Y'

  // MANGLING-DAG: @"\01??_7Y@Test7@@6B@"
};

Y::Y() {}
}

namespace Test8 {

// This is a typical diamond inheritance with a shared 'A' vbase.
struct X : D, C {
  // CHECK-LABEL: VFTable for 'D' in 'Test8::X' (1 entry).
  // CHECK-NEXT: 0 | void D::h()

  // CHECK-LABEL: VFTable for 'A' in 'D' in 'Test8::X' (2 entries).
  // CHECK-NEXT: 0 | void Test8::X::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable indices for 'Test8::X' (1 entry).
  // CHECK-NEXT: via vbtable index 1, vfptr at offset 0
  // CHECK-NEXT: 0 | void Test8::X::f()

  // MANGLING-DAG: @"\01??_7X@Test8@@6BA@@@"
  // MANGLING-DAG: @"\01??_7X@Test8@@6BD@@@"

  virtual void f();
};

X x;
void use(X *obj) { obj->f(); }

// Another diamond inheritance which led to AST crashes.
struct Y : virtual A {};

struct Z : Y, C {
  // CHECK-LABEL: VFTable for 'A' in 'Test8::Y' in 'Test8::Z' (2 entries).
  // CHECK-NEXT: 0 | void Test8::Z::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable indices for 'Test8::Z' (1 entry).
  // CHECK-NEXT: via vbtable index 1, vfptr at offset 0
  // CHECK-NEXT: 0 | void Test8::Z::f()
  virtual void f();
};
Z z;
void use(Z *obj) { obj->f(); }

// Another diamond inheritance which we miscompiled (PR18967).
struct W : virtual A {
  virtual void bar();
};

struct T : W, C {
  // CHECK-LABEL: VFTable for 'Test8::W' in 'Test8::T' (1 entry)
  // CHECK-NEXT: 0 | void Test8::T::bar()

  // CHECK-LABEL: VFTable for 'A' in 'Test8::W' in 'Test8::T' (2 entries)
  // CHECK-NEXT: 0 | void C::f()
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: Thunks for 'void C::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]
  virtual void bar();
  int field;
};
T t;
void use(T *obj) { obj->bar(); }
}

namespace Test9 {

struct X : A { };

struct Y : virtual X {
  // CHECK-LABEL: VFTable for 'Test9::Y' (1 entry).
  // CHECK-NEXT: 0 | void Test9::Y::h()

  // CHECK-LABEL: VFTable for 'A' in 'Test9::X' in 'Test9::Y' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable indices for 'Test9::Y' (1 entry).
  // CHECK-NEXT: 0 | void Test9::Y::h()

  // MANGLING-DAG: @"\01??_7Y@Test9@@6B01@@"
  // MANGLING-DAG: @"\01??_7Y@Test9@@6BX@1@@"

  virtual void h();
};

Y y;
void use(Y *obj) { obj->h(); }

struct Z : Y, virtual B {
  Z();
  // CHECK-LABEL: VFTable for 'Test9::Y' in 'Test9::Z' (1 entry).
  // CHECK-NEXT: 0 | void Test9::Y::h()

  // CHECK-LABEL: VFTable for 'A' in 'Test9::X' in 'Test9::Y' in 'Test9::Z' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable for 'B' in 'Test9::Z' (1 entry).
  // CHECK-NEXT: 0 | void B::g()

  // CHECK-NOT: VFTable indices for 'Test9::Z'

  // MANGLING-DAG: @"\01??_7Z@Test9@@6BX@1@@"
  // MANGLING-DAG: @"\01??_7Z@Test9@@6BY@1@@"

  // MANGLING-DAG: @"\01??_7Z@Test9@@6B@"
};

Z::Z() {}

struct W : Z, D, virtual A, virtual B {
  W();
  // CHECK-LABEL: VFTable for 'Test9::Y' in 'Test9::Z' in 'Test9::W' (1 entry).
  // CHECK-NEXT: 0 | void Test9::Y::h()

  // CHECK-LABEL: VFTable for 'A' in 'Test9::X' in 'Test9::Y' in 'Test9::Z' in 'Test9::W' (2 entries).
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable for 'B' in 'Test9::Z' in 'Test9::W' (1 entry).
  // CHECK-NEXT: 0 | void B::g()

  // CHECK-LABEL: VFTable for 'D' in 'Test9::W' (1 entry).
  // CHECK-NEXT: 0 | void D::h()

  // CHECK-LABEL: VFTable for 'A' in 'D' in 'Test9::W' (2 entries).
  // CHECK-NEXT: 0 | void D::f()
  // CHECK-NEXT:     [this adjustment: -8 non-virtual]
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: Thunks for 'void D::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -8 non-virtual]

  // CHECK-NOT: VFTable indices for 'Test9::W'

  // MANGLING-DAG: @"\01??_7W@Test9@@6BA@@@"
  // MANGLING-DAG: @"\01??_7W@Test9@@6BD@@@"
  // MANGLING-DAG: @"\01??_7W@Test9@@6BX@1@@"

  // MANGLING-DAG: @"\01??_7W@Test9@@6B@"
  // MANGLING-DAG: @"\01??_7W@Test9@@6BY@1@@"
};

W::W() {}

struct T : Z, D, virtual A, virtual B {
  // CHECK-LABEL: VFTable for 'Test9::Y' in 'Test9::Z' in 'Test9::T' (1 entry).
  // CHECK-NEXT: 0 | void Test9::T::h()

  // CHECK-LABEL: VFTable for 'A' in 'Test9::X' in 'Test9::Y' in 'Test9::Z' in 'Test9::T' (2 entries).
  // CHECK-NEXT: 0 | void Test9::T::f()
  // CHECK-NEXT: 1 | void Test9::T::z()

  // CHECK-LABEL: VFTable for 'B' in 'Test9::Z' in 'Test9::T' (1 entry).
  // CHECK-NEXT: 0 | void Test9::T::g()

  // CHECK-LABEL: VFTable for 'D' in 'Test9::T' (1 entry).
  // CHECK-NEXT: 0 | void Test9::T::h()
  // CHECK-NEXT:     [this adjustment: -8 non-virtual]

  // CHECK-LABEL: Thunks for 'void Test9::T::h()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: VFTable for 'A' in 'D' in 'Test9::T' (2 entries).
  // CHECK-NEXT: 0 | void Test9::T::f()
  // CHECK-NEXT:     [this adjustment: -8 non-virtual]
  // CHECK-NEXT: 1 | void Test9::T::z()
  // CHECK-NEXT:     [this adjustment: -8 non-virtual]

  // CHECK-LABEL: Thunks for 'void Test9::T::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: Thunks for 'void Test9::T::z()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: VFTable indices for 'Test9::T' (4 entries).
  // CHECK-NEXT: via vfptr at offset 0
  // CHECK-NEXT: 0 | void Test9::T::h()
  // CHECK-NEXT: via vbtable index 1, vfptr at offset 0
  // CHECK-NEXT: 0 | void Test9::T::f()
  // CHECK-NEXT: 1 | void Test9::T::z()
  // CHECK-NEXT: via vbtable index 2, vfptr at offset 0
  // CHECK-NEXT: 0 | void Test9::T::g()

  // MANGLING-DAG: @"\01??_7T@Test9@@6BA@@@"
  // MANGLING-DAG: @"\01??_7T@Test9@@6BD@@@"
  // MANGLING-DAG: @"\01??_7T@Test9@@6BX@1@@"

  // MANGLING-DAG: @"\01??_7T@Test9@@6B@"
  // MANGLING-DAG: @"\01??_7T@Test9@@6BY@1@@"

  virtual void f();
  virtual void g();
  virtual void h();
  virtual void z();
};

T t;
void use(T *obj) { obj->f(); }
}

namespace Test10 {
struct X : virtual C, virtual A {
  // CHECK-LABEL: VFTable for 'A' in 'C' in 'Test10::X' (2 entries).
  // CHECK-NEXT: 0 | void Test10::X::f()
  // CHECK-NEXT: 1 | void A::z()

  // CHECK-LABEL: VFTable indices for 'Test10::X' (1 entry).
  // CHECK-NEXT: via vbtable index 1, vfptr at offset 0
  // CHECK-NEXT: 0 | void Test10::X::f()
  virtual void f();
};

void X::f() {}
X x;
void use(X *obj) { obj->f(); }
}

namespace Test11 {
struct X : virtual A {};
struct Y { virtual void g(); };

struct Z : virtual X, Y {
  // MANGLING-DAG: @"\01??_7Z@Test11@@6BY@1@@"
  // MANGLING-DAG: @"\01??_7Z@Test11@@6BX@1@@"
};

Z z;

struct W : virtual X, A {};

// Used to crash, PR17748.
W w;
}

namespace Test12 {
struct X : B, A { };

struct Y : X {
  virtual void f();  // Overrides A::f.
};

struct Z : virtual Y {
  // CHECK-LABEL: VFTable for 'A' in 'Test12::X' in 'Test12::Y' in 'Test12::Z' (2 entries).
  // CHECK-NEXT:   0 | void Test12::Y::f()
  // CHECK-NEXT:   1 | void A::z()

  int z;
  // MANGLING-DAG: @"\01??_7Z@Test12@@6BA@@@" = {{.*}}@"\01?f@Y@Test12@@UAEXXZ"
};

struct W : Z {
  // CHECK-LABEL: VFTable for 'A' in 'Test12::X' in 'Test12::Y' in 'Test12::Z' in 'Test12::W' (2 entries).
  // CHECK-NEXT:   0 | void Test12::Y::f()
  // CHECK-NEXT:   1 | void A::z()
  W();

  int w;
  // MANGLING-DAG: @"\01??_7W@Test12@@6BA@@@" = {{.*}}@"\01?f@Y@Test12@@UAEXXZ"
};

W::W() {}
}

namespace vdtors {
struct X {
  virtual ~X();
  virtual void zzz();
};

struct Y : virtual X {
  // CHECK-LABEL: VFTable for 'vdtors::X' in 'vdtors::Y' (2 entries).
  // CHECK-NEXT: 0 | vdtors::Y::~Y() [scalar deleting]
  // CHECK-NEXT: 1 | void vdtors::X::zzz()

  // CHECK-NOT: Thunks for 'vdtors::Y::~Y()'
  virtual ~Y();
};

Y y;
void use(Y *obj) { delete obj; }

struct Z {
  virtual void z();
};

struct W : Z, X {
  // Implicit virtual dtor.
};

struct U : virtual W {
  // CHECK-LABEL: VFTable for 'vdtors::Z' in 'vdtors::W' in 'vdtors::U' (1 entry).
  // CHECK-NEXT: 0 | void vdtors::Z::z()

  // CHECK-LABEL: VFTable for 'vdtors::X' in 'vdtors::W' in 'vdtors::U' (2 entries).
  // CHECK-NEXT: 0 | vdtors::U::~U() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]
  // CHECK-NEXT: 1 | void vdtors::X::zzz()

  // CHECK-LABEL: Thunks for 'vdtors::U::~U()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'vdtors::U' (1 entry).
  // CHECK-NEXT: -- accessible via vbtable index 1, vfptr at offset 4 --
  // CHECK-NEXT: 0 | vdtors::U::~U() [scalar deleting]
  virtual ~U();
};

U u;
void use(U *obj) { delete obj; }

struct V : virtual W {
  // CHECK-LABEL: VFTable for 'vdtors::Z' in 'vdtors::W' in 'vdtors::V' (1 entry).
  // CHECK-NEXT: 0 | void vdtors::Z::z()

  // CHECK-LABEL: VFTable for 'vdtors::X' in 'vdtors::W' in 'vdtors::V' (2 entries).
  // CHECK-NEXT: 0 | vdtors::V::~V() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: -4 non-virtual]
  // CHECK-NEXT: 1 | void vdtors::X::zzz()

  // CHECK-LABEL: Thunks for 'vdtors::V::~V()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'vdtors::V' (1 entry).
  // CHECK-NEXT: -- accessible via vbtable index 1, vfptr at offset 4 --
  // CHECK-NEXT: 0 | vdtors::V::~V() [scalar deleting]
};

V v;
void use(V *obj) { delete obj; }

struct T : virtual X {
  virtual ~T();
};

struct P : T, Y {
  // CHECK-LABEL: VFTable for 'vdtors::X' in 'vdtors::T' in 'vdtors::P' (2 entries).
  // CHECK-NEXT: 0 | vdtors::P::~P() [scalar deleting]
  // CHECK-NEXT: 1 | void vdtors::X::zzz()

  // CHECK-NOT: Thunks for 'vdtors::P::~P()'
  virtual ~P();
};

P p;
void use(P *obj) { delete obj; }

struct Q {
  virtual ~Q();
};

// PR19172: Yet another diamond we miscompiled.
struct R : virtual Q, X {
  // CHECK-LABEL: VFTable for 'vdtors::Q' in 'vdtors::R' (1 entry).
  // CHECK-NEXT: 0 | vdtors::R::~R() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: -8 non-virtual]

  // CHECK-LABEL: Thunks for 'vdtors::R::~R()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: -8 non-virtual]

  // CHECK-LABEL: VFTable for 'vdtors::X' in 'vdtors::R' (2 entries).
  // CHECK-NEXT: 0 | vdtors::R::~R() [scalar deleting]
  // CHECK-NEXT: 1 | void vdtors::X::zzz()

  // CHECK-LABEL: VFTable indices for 'vdtors::R' (1 entry).
  // CHECK-NEXT: 0 | vdtors::R::~R() [scalar deleting]
  virtual ~R();
};

R r;
void use(R *obj) { delete obj; }
}

namespace return_adjustment {

struct X : virtual A {
  virtual void f();
};

struct Y : virtual A, virtual X {
  virtual void f();
};

struct Z {
  virtual A* foo();
};

struct W : Z {
  // CHECK-LABEL: VFTable for 'return_adjustment::Z' in 'return_adjustment::W' (2 entries).
  // CHECK-NEXT: 0 | return_adjustment::X *return_adjustment::W::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct A *'): vbase #1, 0 non-virtual]
  // CHECK-NEXT: 1 | return_adjustment::X *return_adjustment::W::foo()

  // CHECK-LABEL: Thunks for 'return_adjustment::X *return_adjustment::W::foo()' (1 entry).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct A *'): vbase #1, 0 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::W' (1 entry).
  // CHECK-NEXT: 1 | return_adjustment::X *return_adjustment::W::foo()

  virtual X* foo();
};

W w;
void use(W *obj) { obj->foo(); }

struct T : W {
  // CHECK-LABEL: VFTable for 'return_adjustment::Z' in 'return_adjustment::W' in 'return_adjustment::T' (3 entries).
  // CHECK-NEXT: 0 | return_adjustment::Y *return_adjustment::T::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct A *'): vbase #1, 0 non-virtual]
  // CHECK-NEXT: 1 | return_adjustment::Y *return_adjustment::T::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct return_adjustment::X *'): vbase #2, 0 non-virtual]
  // CHECK-NEXT: 2 | return_adjustment::Y *return_adjustment::T::foo()

  // CHECK-LABEL: Thunks for 'return_adjustment::Y *return_adjustment::T::foo()' (2 entries).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct A *'): vbase #1, 0 non-virtual]
  // CHECK-NEXT: 1 | [return adjustment (to type 'struct return_adjustment::X *'): vbase #2, 0 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::T' (1 entry).
  // CHECK-NEXT: 2 | return_adjustment::Y *return_adjustment::T::foo()

  virtual Y* foo();
};

T t;
void use(T *obj) { obj->foo(); }

struct U : virtual A {
  virtual void g();  // adds a vfptr
};

struct V : Z {
  // CHECK-LABEL: VFTable for 'return_adjustment::Z' in 'return_adjustment::V' (2 entries).
  // CHECK-NEXT: 0 | return_adjustment::U *return_adjustment::V::foo()
  // CHECK-NEXT:     [return adjustment (to type 'struct A *'): vbptr at offset 4, vbase #1, 0 non-virtual]
  // CHECK-NEXT: 1 | return_adjustment::U *return_adjustment::V::foo()

  // CHECK-LABEL: Thunks for 'return_adjustment::U *return_adjustment::V::foo()' (1 entry).
  // CHECK-NEXT: 0 | [return adjustment (to type 'struct A *'): vbptr at offset 4, vbase #1, 0 non-virtual]

  // CHECK-LABEL: VFTable indices for 'return_adjustment::V' (1 entry).
  // CHECK-NEXT: 1 | return_adjustment::U *return_adjustment::V::foo()

  virtual U* foo();
};

V v;
void use(V *obj) { obj->foo(); }
}

namespace pr17748 {
struct A {
  virtual void f() {}
};

struct B : virtual A {
  B() {}
};

struct C : virtual B, A {
  C() {}
};
C c;

// MANGLING-DAG: @"\01??_7A@pr17748@@6B@"
// MANGLING-DAG: @"\01??_7B@pr17748@@6B@"
// MANGLING-DAG: @"\01??_7C@pr17748@@6BA@1@@"
// MANGLING-DAG: @"\01??_7C@pr17748@@6BB@1@@"
}

namespace pr19066 {
struct X : virtual B {};

struct Y : virtual X, B {
  Y();
  // CHECK-LABEL: VFTable for 'B' in 'pr19066::X' in 'pr19066::Y' (1 entry).
  // CHECK-NEXT:  0 | void B::g()

  // CHECK-LABEL: VFTable for 'B' in 'pr19066::Y' (1 entry).
  // CHECK-NEXT:  0 | void B::g()
};

Y::Y() {}
}

namespace pr19240 {
struct A {
  virtual void c();
};

struct B : virtual A {
  virtual void c();
};

struct C { };

struct D : virtual A, virtual C, B {};

D obj;

// Each MDC only has one vftable.

// MANGLING-DAG: @"\01??_7D@pr19240@@6B@"
// MANGLING-DAG: @"\01??_7A@pr19240@@6B@"
// MANGLING-DAG: @"\01??_7B@pr19240@@6B@"

}

namespace pr19408 {
// This test is a non-vtordisp version of the reproducer for PR19408.
struct X : virtual A {
  int x;
};

struct Y : X {
  virtual void f();
  int y;
};

struct Z : Y {
  // CHECK-LABEL: VFTable for 'A' in 'pr19408::X' in 'pr19408::Y' in 'pr19408::Z' (2 entries).
  // CHECK-NEXT:   0 | void pr19408::Y::f()
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   1 | void A::z()

  Z();
  int z;
  // MANGLING-DAG: @"\01??_7Z@pr19408@@6B@" = {{.*}}@"\01?f@Y@pr19408@@W3AEXXZ"
};

Z::Z() {}

struct W : B, Y {
  // CHECK-LABEL: VFTable for 'A' in 'pr19408::X' in 'pr19408::Y' in 'pr19408::W' (2 entries).
  // CHECK-NEXT:   0 | void pr19408::Y::f()
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]
  // CHECK-NEXT:   1 | void A::z()

  W();
  int w;
  // MANGLING-DAG: @"\01??_7W@pr19408@@6BY@1@@" = {{.*}}@"\01?f@Y@pr19408@@W3AEXXZ"
};

W::W() {}
}

namespace Test13 {
struct A {
  virtual void f();
};
struct __declspec(dllexport) B : virtual A {
  virtual void f() = 0;
  // MANGLING-DAG: @"\01??_7B@Test13@@6B@" = weak_odr dllexport unnamed_addr constant { [1 x i8*] } { [1 x i8*] [i8* bitcast (void ()* @_purecall to i8*)] }
};
}

namespace pr21031_1 {
// This ordering of base specifiers regressed in r202425.
struct A { virtual void f(void); };
struct B : virtual A { virtual void g(void); };
struct C : virtual A, B { C(); };
C::C() {}

// CHECK-LABEL: VFTable for 'pr21031_1::A' in 'pr21031_1::B' in 'pr21031_1::C' (1 entry)
// CHECK-NEXT:   0 | void pr21031_1::A::f()

// CHECK-LABEL: VFTable for 'pr21031_1::B' in 'pr21031_1::C' (1 entry)
// CHECK-NEXT:   0 | void pr21031_1::B::g()

// MANGLING-DAG: @"\01??_7C@pr21031_1@@6BB@1@@" = {{.*}} constant { [1 x i8*] }
// MANGLING-DAG: @"\01??_7C@pr21031_1@@6B@" = {{.*}} constant { [1 x i8*] }
}

namespace pr21031_2 {
struct A { virtual void f(void); };
struct B : virtual A { virtual void g(void); };
struct C : B, virtual A { C(); };
C::C() {}

// CHECK-LABEL: VFTable for 'pr21031_2::B' in 'pr21031_2::C' (1 entry)
// CHECK-NEXT:   0 | void pr21031_2::B::g()

// CHECK-LABEL: VFTable for 'pr21031_2::A' in 'pr21031_2::B' in 'pr21031_2::C' (1 entry)
// CHECK-NEXT:   0 | void pr21031_2::A::f()

// MANGLING-DAG: @"\01??_7C@pr21031_2@@6BA@1@@" = {{.*}} constant { [1 x i8*] }
// MANGLING-DAG: @"\01??_7C@pr21031_2@@6BB@1@@" = {{.*}} constant { [1 x i8*] }
}

namespace pr21062_1 {
struct A { virtual void f(); };
struct B {};
struct C : virtual B {};
struct D : virtual C, virtual B, virtual A { D();};
D::D() {}

// CHECK-LABEL: VFTable for 'pr21062_1::A' in 'pr21062_1::D' (1 entry)
// CHECK-NEXT:   0 | void pr21062_1::A::f()

// MANGLING-DAG: @"\01??_7D@pr21062_1@@6B@" = {{.*}} constant { [1 x i8*] }
}

namespace pr21062_2 {
struct A { virtual void f(); };
struct B {};
struct C : virtual B {};
struct D : C, virtual B, virtual A { D(); };
D::D() {}

// CHECK-LABEL: VFTable for 'pr21062_2::A' in 'pr21062_2::D' (1 entry)
// CHECK-NEXT:   0 | void pr21062_2::A::f()

// MANGLING-DAG: @"\01??_7D@pr21062_2@@6B@" = {{.*}} constant { [1 x i8*] }
}

namespace pr21064 {
struct A {};
struct B { virtual void f(); };
struct C : virtual A, virtual B {};
struct D : virtual A, virtual C { D(); };
D::D() {}
// CHECK-LABEL: VFTable for 'pr21064::B' in 'pr21064::C' in 'pr21064::D' (1 entry)
// CHECK-NEXT:   0 | void pr21064::B::f()

// MANGLING-DAG: @"\01??_7D@pr21064@@6B@" = {{.*}} constant { [1 x i8*] }
}
