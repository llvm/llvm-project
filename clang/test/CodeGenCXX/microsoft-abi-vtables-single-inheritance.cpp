// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -fdump-vtable-layouts -o %t.ll > %t
// RUN: FileCheck --check-prefix=EMITS-VFTABLE %s < %t.ll
// RUN: FileCheck --check-prefix=NO-VFTABLE %s < %t.ll
// RUN: FileCheck %s < %t

struct A {
  // CHECK-LABEL: VFTable for 'A' (3 entries)
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::g()
  // CHECK-NEXT: 2 | void A::h()
  // CHECK-LABEL: VFTable indices for 'A' (3 entries)
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::g()
  // CHECK-NEXT: 2 | void A::h()

  virtual void f();
  virtual void g();
  virtual void h();
  int ia;
};
A a;
// EMITS-VFTABLE-DAG: @"??_7A@@6B@" = linkonce_odr unnamed_addr constant { [3 x ptr] }
void use(A *obj) { obj->f(); }

struct B : A {
  // CHECK-LABEL: VFTable for 'A' in 'B' (5 entries)
  // CHECK-NEXT: 0 | void B::f()
  // CHECK-NEXT: 1 | void A::g()
  // CHECK-NEXT: 2 | void A::h()
  // CHECK-NEXT: 3 | void B::i()
  // CHECK-NEXT: 4 | void B::j()
  // CHECK-LABEL: VFTable indices for 'B' (3 entries)
  // CHECK-NEXT: 0 | void B::f()
  // CHECK-NEXT: 3 | void B::i()
  // CHECK-NEXT: 4 | void B::j()

  virtual void f();  // overrides A::f()
  virtual void i();
  virtual void j();
};
B b;
// EMITS-VFTABLE-DAG: @"??_7B@@6B@" = linkonce_odr unnamed_addr constant { [5 x ptr] }
void use(B *obj) { obj->f(); }

struct C {
  // CHECK-LABEL: VFTable for 'C' (2 entries)
  // CHECK-NEXT: 0 | C::~C() [scalar deleting]
  // CHECK-NEXT: 1 | void C::f()
  // CHECK-LABEL: VFTable indices for 'C' (2 entries).
  // CHECK-NEXT: 0 | C::~C() [scalar deleting]
  // CHECK-NEXT: 1 | void C::f()

  virtual ~C();
  virtual void f();
};
void C::f() {}
// NO-VFTABLE-NOT: @"??_7C@@6B@"
void use(C *obj) { obj->f(); }

struct D {
  // CHECK-LABEL: VFTable for 'D' (2 entries)
  // CHECK-NEXT: 0 | void D::f()
  // CHECK-NEXT: 1 | D::~D() [scalar deleting]
  // CHECK-LABEL: VFTable indices for 'D' (2 entries)
  // CHECK-NEXT: 0 | void D::f()
  // CHECK-NEXT: 1 | D::~D() [scalar deleting]

  virtual void f();
  virtual ~D();
};
D d;
// EMITS-VFTABLE-DAG: @"??_7D@@6B@" = linkonce_odr unnamed_addr constant { [2 x ptr] }
void use(D *obj) { obj->f(); }

struct E : A {
  // CHECK-LABEL: VFTable for 'A' in 'E' (5 entries)
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::g()
  // CHECK-NEXT: 2 | void A::h()
  // CHECK-NEXT: 3 | E::~E() [scalar deleting]
  // CHECK-NEXT: 4 | void E::i()
  // CHECK-LABEL: VFTable indices for 'E' (2 entries).
  // CHECK-NEXT: 3 | E::~E() [scalar deleting]
  // CHECK-NEXT: 4 | void E::i()

  // ~E would be the key method, but it isn't used, and MS ABI has no key
  // methods.
  virtual ~E();
  virtual void i();
};
void E::i() {}
// NO-VFTABLE-NOT: @"??_7E@@6B@"
void use(E *obj) { obj->i(); }

struct F : A {
  // CHECK-LABEL: VFTable for 'A' in 'F' (5 entries)
  // CHECK-NEXT: 0 | void A::f()
  // CHECK-NEXT: 1 | void A::g()
  // CHECK-NEXT: 2 | void A::h()
  // CHECK-NEXT: 3 | void F::i()
  // CHECK-NEXT: 4 | F::~F() [scalar deleting]
  // CHECK-LABEL: VFTable indices for 'F' (2 entries).
  // CHECK-NEXT: 3 | void F::i()
  // CHECK-NEXT: 4 | F::~F() [scalar deleting]

  virtual void i();
  virtual ~F();
};
F f;
// EMITS-VFTABLE-DAG: @"??_7F@@6B@" = linkonce_odr unnamed_addr constant { [5 x ptr] }
void use(F *obj) { obj->i(); }

struct G : E {
  // CHECK-LABEL: VFTable for 'A' in 'E' in 'G' (6 entries)
  // CHECK-NEXT: 0 | void G::f()
  // CHECK-NEXT: 1 | void A::g()
  // CHECK-NEXT: 2 | void A::h()
  // CHECK-NEXT: 3 | G::~G() [scalar deleting]
  // CHECK-NEXT: 4 | void E::i()
  // CHECK-NEXT: 5 | void G::j()
  // CHECK-LABEL: VFTable indices for 'G' (3 entries).
  // CHECK-NEXT: 0 | void G::f()
  // CHECK-NEXT: 3 | G::~G() [scalar deleting]
  // CHECK-NEXT: 5 | void G::j()

  virtual void f();  // overrides A::f()
  virtual ~G();
  virtual void j();
};
void G::j() {}
// NO-VFTABLE-NOT: @"??_7G@@6B@"
void use(G *obj) { obj->j(); }

// Test that the usual Itanium-style key method does not emit a vtable.
struct H {
  virtual void f();
};
void H::f() {}
// NO-VFTABLE-NOT: @"??_7H@@6B@"

struct Empty { };

struct I : Empty {
  // CHECK-LABEL: VFTable for 'I' (2 entries)
  // CHECK-NEXT: 0 | void I::f()
  // CHECK-NEXT: 1 | void I::g()
  virtual void f();
  virtual void g();
};

I i;
void use(I *obj) { obj->f(); }

struct J {
  // CHECK-LABEL: VFTable for 'J' (6 entries)
  // CHECK-NEXT: 0 | void J::foo(long)
  // CHECK-NEXT: 1 | void J::foo(int)
  // CHECK-NEXT: 2 | void J::foo(short)
  // CHECK-NEXT: 3 | void J::bar(long)
  // CHECK-NEXT: 4 | void J::bar(int)
  // CHECK-NEXT: 5 | void J::bar(short)
  virtual void foo(short);
  virtual void bar(short);
  virtual void foo(int);
  virtual void bar(int);
  virtual void foo(long);
  virtual void bar(long);
};

J j;
void use(J *obj) { obj->foo(42); }

struct K : J {
  // CHECK-LABEL: VFTable for 'J' in 'K' (9 entries)
  // CHECK-NEXT: 0 | void J::foo(long)
  // CHECK-NEXT: 1 | void J::foo(int)
  // CHECK-NEXT: 2 | void J::foo(short)
  // CHECK-NEXT: 3 | void J::bar(long)
  // CHECK-NEXT: 4 | void J::bar(int)
  // CHECK-NEXT: 5 | void J::bar(short)
  // CHECK-NEXT: 6 | void K::bar(double)
  // CHECK-NEXT: 7 | void K::bar(float)
  // CHECK-NEXT: 8 | void K::foo(float)
  virtual void bar(float);
  virtual void foo(float);
  virtual void bar(double);
};

K k;
void use(K *obj) { obj->foo(42.0f); }

struct L : J {
  // CHECK-LABEL: VFTable for 'J' in 'L' (9 entries)
  // CHECK-NEXT: 0 | void J::foo(long)
  // CHECK-NEXT: 1 | void L::foo(int)
  // CHECK-NEXT: 2 | void J::foo(short)
  // CHECK-NEXT: 3 | void J::bar(long)
  // CHECK-NEXT: 4 | void J::bar(int)
  // CHECK-NEXT: 5 | void J::bar(short)
  // CHECK-NEXT: 6 | void L::foo(float)
  // CHECK-NEXT: 7 | void L::bar(double)
  // CHECK-NEXT: 8 | void L::bar(float)

  // This case is interesting. Since the J::foo(int) override is the first method in
  // the class, foo(float) precedes the bar(double) and bar(float) in the vftable.
  virtual void foo(int);
  virtual void bar(float);
  virtual void foo(float);
  virtual void bar(double);
};

L l;
void use(L *obj) { obj->foo(42.0f); }

struct M : J {
  // CHECK-LABEL: VFTable for 'J' in 'M' (11 entries)
  // CHECK-NEXT:  0 | void J::foo(long)
  // CHECK-NEXT:  1 | void M::foo(int)
  // CHECK-NEXT:  2 | void J::foo(short)
  // CHECK-NEXT:  3 | void J::bar(long)
  // CHECK-NEXT:  4 | void J::bar(int)
  // CHECK-NEXT:  5 | void J::bar(short)
  // CHECK-NEXT:  6 | void M::foo(float)
  // CHECK-NEXT:  7 | void M::spam(long)
  // CHECK-NEXT:  8 | void M::spam(int)
  // CHECK-NEXT:  9 | void M::bar(double)
  // CHECK-NEXT: 10 | void M::bar(float)

  virtual void foo(int);
  virtual void spam(int);
  virtual void bar(float);
  virtual void bar(double);
  virtual void foo(float);
  virtual void spam(long);
};

M m;
void use(M *obj) { obj->foo(42.0f); }

struct N {
  // CHECK-LABEL: VFTable for 'N' (4 entries)
  // CHECK-NEXT: 0 | void N::operator+(int)
  // CHECK-NEXT: 1 | void N::operator+(short)
  // CHECK-NEXT: 2 | void N::operator*(int)
  // CHECK-NEXT: 3 | void N::operator*(short)
  virtual void operator+(short);
  virtual void operator*(short);
  virtual void operator+(int);
  virtual void operator*(int);
};

N n;
void use(N *obj) { obj->operator+(42); }

struct O { virtual A *f(); };
struct P : O { virtual B *f(); };
P p;
void use(O *obj) { obj->f(); }
void use(P *obj) { obj->f(); }
// CHECK-LABEL: VFTable for 'O' (1 entry)
// CHECK-NEXT: 0 | A *O::f()

// CHECK-LABEL: VFTable for 'O' in 'P' (1 entry)
// CHECK-NEXT: 0 | B *P::f()

struct Q {
  // CHECK-LABEL: VFTable for 'Q' (2 entries)
  // CHECK-NEXT: 0 | void Q::foo(int)
  // CHECK-NEXT: 1 | void Q::bar(int)
  void foo(short);
  void bar(short);
  virtual void bar(int);
  virtual void foo(int);
};

Q q;
void use(Q *obj) { obj->foo(42); }

// Inherited non-virtual overloads don't participate in the ordering.
struct R : Q {
  // CHECK-LABEL: VFTable for 'Q' in 'R' (4 entries)
  // CHECK-NEXT: 0 | void Q::foo(int)
  // CHECK-NEXT: 1 | void Q::bar(int)
  // CHECK-NEXT: 2 | void R::bar(long)
  // CHECK-NEXT: 3 | void R::foo(long)
  virtual void bar(long);
  virtual void foo(long);
};

R r;
void use(R *obj) { obj->foo(42l); }

struct S {
  // CHECK-LABEL: VFTable for 'S' (1 entry).
  // CHECK-NEXT:   0 | void S::f() [deleted]
  virtual void f() = delete;
  S();
  // EMITS-VFTABLE-DAG: @"??_7S@@6B@" = linkonce_odr unnamed_addr constant { [1 x ptr] } { [1 x ptr] [ptr @_purecall] }
};

S::S() {}

struct T {
  struct U {};
};
struct V : T {
  // CHECK-LABEL: VFTable for 'V' (2 entries).
  // CHECK-NEXT:   0 | void V::U()
  // CHECK-NEXT:   1 | void V::f()
  using T::U;
  virtual void f();
  virtual void U();
  V();
};

V::V() {}
