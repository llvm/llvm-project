// RUN: %clang_cc1 -fno-rtti -emit-llvm -fdump-vtable-layouts %s -o %t.ll -triple=i386-pc-win32 > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck --check-prefix=MANGLING %s < %t.ll

// For now, just make sure x86_64 doesn't crash.
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -fdump-vtable-layouts %s -triple=x86_64-pc-win32 > /dev/null

struct V1 {
  virtual void f();
  virtual ~V1();
};

struct V2 {
  virtual void f();
  virtual ~V2();
  int v;
};

struct Z {
  virtual void g();
  virtual ~Z();
  int x;
};

struct V3 : Z, V2 {
};

struct V4 : Z, V1, V2 {
  int y;
};

void use_somewhere_else(void*);

namespace simple {
// In case of a single-layer virtual inheritance, the "this" adjustment for a
// virtual method is done statically:
//   struct A {
//     virtual void f();  // Expects "(A*)this" in ECX
//   };
//   struct B : virtual A {
//     virtual void f();  // Expects "(char*)(B*)this + 12" in ECX
//     virtual ~B();      // Might call f()
//   };
//
// If a class overrides a virtual function of its base and has a non-trivial
// ctor/dtor that call(s) the virtual function (or may escape "this" to some
// code that might call it), a virtual adjustment might be needed in case the
// current class layout and the most derived class layout are different.
// This is done using vtordisp thunks.
//
// A simple vtordisp{x,y} thunk for Method@Class is something like:
//   sub  ecx, [ecx+x]  // apply the vtordisp adjustment
//   sub  ecx, y        // apply the subobject adjustment, if needed.
//   jmp Method@Class

struct A : virtual V1 {
  // CHECK-LABEL: VFTable for 'V1' in 'simple::A' (2 entries).
  // CHECK-NEXT: 0 | void simple::A::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]
  // CHECK-NEXT: 1 | simple::A::~A() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: Thunks for 'simple::A::~A()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::A::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, 0 non-virtual]

  virtual void f();
  // MANGLING-DAG: @"?f@A@simple@@$4PPPPPPPM@A@AEXXZ"

  virtual ~A();
  // MANGLING-DAG: @"??_EA@simple@@$4PPPPPPPM@A@AEPAXI@Z"
};

A a;
void use(A *obj) { obj->f(); }

struct B : virtual V3 {
  // CHECK-LABEL: VFTable for 'Z' in 'V3' in 'simple::B' (2 entries).
  // CHECK-NEXT: 0 | void Z::g()
  // CHECK-NEXT: 1 | simple::B::~B() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: Thunks for 'simple::B::~B()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: VFTable for 'V2' in 'V3' in 'simple::B' (2 entries).
  // CHECK-NEXT: 0 | void simple::B::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -12, 0 non-virtual]
  // CHECK-NEXT: 1 | simple::B::~B() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -12, -8 non-virtual]

  // CHECK-LABEL: Thunks for 'simple::B::~B()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -12, -8 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::B::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -12, 0 non-virtual]

  // FIXME: The vtordisp thunk should only get emitted for a constructor
  // if "this" leaves scope.
  B() { use_somewhere_else(this); }

  virtual void f();
  // MANGLING-DAG: @"?f@B@simple@@$4PPPPPPPE@A@AEXXZ"

  // Has an implicit destructor.
  // MANGLING-DAG: @"??_EB@simple@@$4PPPPPPPE@7AEPAXI@Z"
  // MANGLING-DAG: @"??_EB@simple@@$4PPPPPPPM@A@AEPAXI@Z"
};

B b;
void use(B *obj) { obj->f(); }

struct C : virtual V4 {
  // CHECK-LABEL: VFTable for 'Z' in 'V4' in 'simple::C' (2 entries).
  // CHECK-NEXT: 0 | void Z::g()
  // CHECK-NEXT: 1 | simple::C::~C() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: Thunks for 'simple::C::~C()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: VFTable for 'V1' in 'V4' in 'simple::C' (2 entries).
  // CHECK-NEXT: 0 | void simple::C::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -12, 0 non-virtual]
  // CHECK-NEXT: 1 | simple::C::~C() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -12, -8 non-virtual]

  // CHECK-LABEL: Thunks for 'simple::C::~C()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -12, -8 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::C::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -12, 0 non-virtual]

  // CHECK-LABEL: VFTable for 'V2' in 'V4' in 'simple::C' (2 entries).
  // CHECK-NEXT: 0 | void simple::C::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -16, -4 non-virtual]
  // CHECK-NEXT: 1 | simple::C::~C() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -16, -12 non-virtual]

  // CHECK-LABEL: Thunks for 'simple::C::~C()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -16, -12 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::C::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -16, -4 non-virtual]

  int x;
  virtual void f();
  // MANGLING-DAG: @"?f@C@simple@@$4PPPPPPPA@3AEXXZ"
  // MANGLING-DAG: @"?f@C@simple@@$4PPPPPPPE@A@AEXXZ"
  virtual ~C();
  // MANGLING-DAG: @"??_EC@simple@@$4PPPPPPPA@M@AEPAXI@Z"
  // MANGLING-DAG: @"??_EC@simple@@$4PPPPPPPE@7AEPAXI@Z"
  // MANGLING-DAG: @"??_EC@simple@@$4PPPPPPPM@A@AEPAXI@Z"
};

C c;
void use(C *obj) { obj->f(); }

class D : B {
  // CHECK-LABEL: VFTable for 'V2' in 'V3' in 'simple::B' in 'simple::D' (2 entries).
  // CHECK-NEXT: 0 | void simple::B::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -12, -4 non-virtual]
  // CHECK-NEXT: 1 | simple::D::~D() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -12, -8 non-virtual]
  D();
  int z;

  // MANGLING-DAG: @"?f@B@simple@@$4PPPPPPPE@3AEXXZ"
};

D::D() {}

struct E : V3 {
  virtual void f();
};

struct F : virtual E {
  // CHECK-LABEL: VFTable for 'Z' in 'V3' in 'simple::E' in 'simple::F' (2 entries).
  // CHECK-NEXT:   0 | void simple::F::g()
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, 0 non-virtual]
  // CHECK-NEXT:   1 | simple::F::~F() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: VFTable for 'V2' in 'V3' in 'simple::E' in 'simple::F' (2 entries).
  // CHECK-NEXT:   0 | void simple::E::f()
  // CHECK-NEXT:   1 | simple::F::~F() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: vtordisp at -12, -8 non-virtual]

  F();
  virtual void g();  // Force a vtordisp.
  int f;

  // MANGLING-DAG: @"?g@F@simple@@$4PPPPPPPM@A@AEXXZ"{{.*}}??_EF@simple@@$4PPPPPPPM@A@AEPAXI@Z
  // MANGLING-DAG: ?f@E@simple@@UAEXXZ{{.*}}??_EF@simple@@$4PPPPPPPE@7AEPAXI@Z
};

F::F() {}

struct G : F {
  // CHECK-LABEL: VFTable for 'Z' in 'V3' in 'simple::E' in 'simple::F' in 'simple::G' (2 entries).
  // CHECK-NEXT:   0 | void simple::F::g()
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, -4 non-virtual]
  // CHECK-NEXT:   1 | simple::G::~G() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: VFTable for 'V2' in 'V3' in 'simple::E' in 'simple::F' in 'simple::G' (2 entries).
  // CHECK-NEXT:   0 | void simple::E::f()
  // CHECK-NEXT:   1 | simple::G::~G() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: vtordisp at -12, -8 non-virtual]

  G();
  int g;

  // MANGLING-DAG: @"?g@F@simple@@$4PPPPPPPM@3AEXXZ"{{.*}}@"??_EG@simple@@$4PPPPPPPM@A@AEPAXI@Z"
  // MANGLING-DAG: @"?f@E@simple@@UAEXXZ"{{.*}}@"??_EG@simple@@$4PPPPPPPE@7AEPAXI@Z"
};

G::G() {}
}

namespace extended {
// If a virtual function requires vtordisp adjustment and the final overrider
// is defined in another virtual base of the most derived class,
// we need to know two vbase offsets.
// In this case, we should use the extended form of vtordisp thunks, called
// vtordispex thunks.
//
// vtordispex{x,y,z,w} thunk for Method@Class is something like:
//   sub  ecx, [ecx+z]  // apply the vtordisp adjustment
//   sub  ecx, x        // jump to the vbptr of the most derived class
//   mov  eax, [ecx]    // load the vbtable address
//   add  ecx, [eax+y]  // lookup the final overrider's vbase offset
//   add  ecx, w        // apphy the subobject offset if needed
//   jmp Method@Class

struct A : virtual simple::A {
  // CHECK-LABEL: VFTable for 'V1' in 'simple::A' in 'extended::A' (2 entries).
  // CHECK-NEXT: 0 | void simple::A::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]
  // CHECK-NEXT: 1 | extended::A::~A() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::A::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // `vtordispex{8,8,4294967292,8}'
  // MANGLING-DAG: @"?f@A@simple@@$R477PPPPPPPM@7AEXXZ"

  virtual ~A();
  // vtordisp{4294967292,0}
  // MANGLING-DAG: @"??_EA@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

A a;
void use(A *obj) { delete obj; }

struct B : virtual simple::A {
  // This class has an implicit dtor.  Vdtors don't require vtordispex thunks
  // as the most derived class always has an implicit dtor,
  // which is a final overrider.

  // CHECK-LABEL: VFTable for 'V1' in 'simple::A' in 'extended::B' (2 entries).
  //  ...
  // CHECK: 1 | extended::B::~B() [scalar deleting]
  // CHECK-NEXT: [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::A::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // vtordisp{4294967292,0}
  // MANGLING-DAG: @"??_EB@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

B b;
void use(B *obj) { delete obj; }

struct C : virtual simple::A {
  // CHECK-LABEL: VFTable for 'V1' in 'simple::A' in 'extended::C' (2 entries).
  // CHECK-NEXT: 0 | void simple::A::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, vbptr at 12 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::A::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, vbptr at 12 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // `vtordispex{12,8,4294967292,8}'
  // MANGLING-DAG: @"?f@A@simple@@$R4M@7PPPPPPPM@7AEXXZ"
  int x;
  virtual ~C();
  // MANGLING-DAG: @"??_EC@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

C c;
void use(C *obj) { delete obj; }

struct D : virtual V2 {
  virtual void f();
  virtual ~D();
  int x;
};

struct E : virtual D {
  // CHECK-LABEL: VFTable for 'V2' in 'extended::D' in 'extended::E' (2 entries).
  // CHECK-NEXT: 0 | void extended::D::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 12 non-virtual]

  // CHECK-LABEL: Thunks for 'void extended::D::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 12 non-virtual]

  // `vtordispex{8,8,4294967292,12}'
  // MANGLING-DAG: @"?f@D@extended@@$R477PPPPPPPM@M@AEXXZ"

  virtual ~E();
  // MANGLING-DAG: @"??_EE@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

E e;
void use(E *obj) { delete obj; } 

struct F : virtual Z, virtual D {
  // CHECK-LABEL: VFTable for 'V2' in 'extended::D' in 'extended::F' (2 entries).
  // CHECK-NEXT: 0 | void extended::D::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, vbptr at 20 to the left,
  // CHECK-NEXT:      vboffset at 12 in the vbtable, 12 non-virtual]

  // CHECK-LABEL: Thunks for 'void extended::D::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, vbptr at 20 to the left,
  // CHECK-NEXT:      vboffset at 12 in the vbtable, 12 non-virtual]

  // `vtordispex{20,12,4294967292,12}'
  // MANGLING-DAG: @"?f@D@extended@@$R4BE@M@PPPPPPPM@M@AEXXZ"
  int x;
  virtual ~F();
  // MANGLING-DAG: @"??_EF@extended@@$4PPPPPPPM@M@AEPAXI@Z"
};

F f;
void use(F *obj) { delete obj; }

struct G : virtual simple::A {
  // CHECK-LABEL: VFTable for 'extended::G' (1 entry).
  // CHECK-NEXT: 0 | void extended::G::g()

  // CHECK-LABEL: VFTable for 'V1' in 'simple::A' in 'extended::G' (2 entries).
  // CHECK-NEXT: 0 | void simple::A::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]
  // CHECK-NEXT: 1 | extended::G::~G() [scalar deleting]
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, 0 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::A::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // Emits a G's own vfptr, thus moving the vbptr in the layout.
  virtual void g();

  virtual ~G();
  // vtordisp{4294967292,0}
  // MANGLING-DAG: @"??_EG@extended@@$4PPPPPPPM@A@AEPAXI@Z"
};

G g;
void use(G *obj) { obj->g(); }

struct H : Z, A {
  // CHECK-LABEL: VFTable for 'Z' in 'extended::H' (2 entries).
  // CHECK-NEXT: 0 | void Z::g()
  // CHECK-NEXT: 1 | extended::H::~H() [scalar deleting]

  // CHECK-LABEL: VFTable for 'V1' in 'simple::A' in 'extended::A' in 'extended::H' (2 entries).
  // CHECK-NEXT: 0 | void simple::A::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::A::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -4, vbptr at 8 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 8 non-virtual]

  // MANGLING-DAG: @"?f@A@simple@@$R477PPPPPPPM@7AEXXZ"
  // MANGLING-DAG: @"??_EH@extended@@$4PPPPPPPM@BA@AEPAXI@Z"
};

H h;
void use(H *obj) { delete obj; }
}

namespace pr17738 {
// These classes should have vtordispex thunks but MSVS CL miscompiles them.
// Just do the right thing.

struct A : virtual simple::B {
  // CHECK-LABEL: VFTable for 'V2' in 'V3' in 'simple::B' in 'pr17738::A' (2 entries).
  // CHECK-NEXT: 0 | void simple::B::f()
  // CHECK-NEXT:     [this adjustment: vtordisp at -12, vbptr at 20 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 16 non-virtual]

  // CHECK-LABEL: Thunks for 'void simple::B::f()' (1 entry).
  // CHECK-NEXT: 0 | [this adjustment: vtordisp at -12, vbptr at 20 to the left,
  // CHECK-NEXT:      vboffset at 8 in the vbtable, 16 non-virtual]

  // MANGLING-DAG: @"?f@B@simple@@$R4BE@7PPPPPPPE@BA@AEXXZ"
  int a;
  virtual ~A();
};

A a;
void use(A *obj) { delete obj; }
}

namespace pr19408 {
// In this test, the vptr used to vcall D::f() is located in the A vbase.
// The offset of A in different in C and D, so the D vtordisp thunk should
// adjust "this" so C::f gets the right value.
struct A {
  A();
  virtual void f();
  int a;
};

struct B : virtual A {
  B();
  int b;
};

struct C : B {
  C();
  virtual void f();
  int c;
};

struct D : C {
  // CHECK-LABEL: VFTable for 'pr19408::A' in 'pr19408::B' in 'pr19408::C' in 'pr19408::D' (1 entry).
  // CHECK-NEXT:   0 | void pr19408::C::f()
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, -4 non-virtual]

  // MANGLING-DAG: @"?f@C@pr19408@@$4PPPPPPPM@3AEXXZ"
  D();
  int d;
};

D::D() {}
}

namespace access {
struct A {
  virtual ~A();
protected:
  virtual void prot();
private:
  virtual void priv();
};

struct B : virtual A {
  virtual ~B();
protected:
  virtual void prot();
  // MANGLING-DAG: @"?prot@B@access@@$2PPPPPPPM@A@AEXXZ"
private:
  virtual void priv();
  // MANGLING-DAG: @"?priv@B@access@@$0PPPPPPPM@A@AEXXZ"
};

B b;

struct C : virtual B {
  virtual ~C();

  // MANGLING-DAG: @"?prot@B@access@@$R277PPPPPPPM@7AEXXZ"
  // MANGLING-DAG: @"?priv@B@access@@$R077PPPPPPPM@7AEXXZ"
};

C c;
}

namespace pr19505 {
struct A {
  virtual void f();
  virtual void z();
};

struct B : A {
  virtual void f();
};

struct C : A, B {
  virtual void g();
};

struct X : B, virtual C {
  X() {}
  virtual void g();

  // CHECK-LABEL: VFTable for 'pr19505::A' in 'pr19505::B' in 'pr19505::C' in 'pr19505::X' (2 entries).
  // CHECK-NEXT:   0 | void pr19505::B::f()
  // CHECK-NEXT:   1 | void pr19505::A::z()

  // MANGLING-DAG: @"??_7X@pr19505@@6BB@1@@" = {{.*}}@"?f@B@pr19505@@UAEXXZ"
} x;

void build_vftable(X *obj) { obj->g(); }
}

namespace pr19506 {
struct A {
  virtual void f();
  virtual void g();
};

struct B : A {
  virtual void f();
};

struct C : B {};

struct X : C, virtual B {
  virtual void g();
  X() {}

  // CHECK-LABEL: VFTable for 'pr19506::A' in 'pr19506::B' in 'pr19506::X' (2 entries).
  // CHECK-NEXT:   0 | void pr19506::B::f()
  // CHECK-NEXT:   1 | void pr19506::X::g()
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, -12 non-virtual]

  // MANGLING-DAG: @"??_7X@pr19506@@6BB@1@@" = {{.*}}@"?f@B@pr19506@@UAEXXZ"
} x;

void build_vftable(X *obj) { obj->g(); }
}

namespace pr19519 {
// VS2013 CL miscompiles this, just make sure we don't regress.

struct A {
  virtual void f();
  virtual void g();
};

struct B : virtual A {
  virtual void f();
  B();
};

struct C : virtual A {
  virtual void g();
};

struct X : B, C {
  X();

  // CHECK-LABEL: VFTable for 'pr19519::A' in 'pr19519::B' in 'pr19519::X' (2 entries).
  // CHECK-NEXT:   0 | void pr19519::B::f()
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, -4 non-virtual]
  // CHECK-NEXT:   1 | void pr19519::C::g()
  // CHECK-NEXT:       [this adjustment: vtordisp at -4, -4 non-virtual]

  // MANGLING-DAG: @"??_7X@pr19519@@6B@" = {{.*}}@"?g@C@pr19519@@$4PPPPPPPM@3AEXXZ"
};

X::X() {}
}
