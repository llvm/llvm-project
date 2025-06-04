// RUN: %clang_cc1 %s -fno-rtti -triple=i386-pc-win32 -emit-llvm -o %t.ll -fdump-vtable-layouts >%t
// RUN: FileCheck %s < %t

struct A {
  virtual ~A();
  virtual void z1();
};

struct B {
  virtual ~B();
};

struct C : A, B {
  // CHECK-LABEL: VFTable for 'A' in 'C' (2 entries).
  // CHECK-NEXT:   0 | C::~C() [scalar deleting]
  // CHECK-NEXT:   1 | void A::z1()

  // CHECK-LABEL: VFTable for 'B' in 'C' (1 entry).
  // CHECK-NEXT:   0 | C::~C() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'C::~C()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'C' (1 entry).
  // CHECK-NEXT:   0 | C::~C() [scalar deleting]
  virtual ~C();
};

void build_vftable(C *obj) { delete obj; }

struct D {
  // No virtual destructor here!
  virtual void z4();
};

struct E : D, B {
  // Implicit virtual dtor here!

  // CHECK-LABEL: VFTable for 'D' in 'E' (1 entry).
  // CHECK-NEXT:   0 | void D::z4()

  // CHECK-LABEL: VFTable for 'B' in 'E' (1 entry).
  // CHECK-NEXT:   0 | E::~E() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'E::~E()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'E' (1 entry).
  // CHECK-NEXT:   -- accessible via vfptr at offset 4 --
  // CHECK-NEXT:   0 | E::~E() [scalar deleting]
};

void build_vftable(E *obj) { delete obj; }

struct F : D, B {
  // Implicit virtual dtor here!

  // CHECK-LABEL: VFTable for 'D' in 'F' (1 entry).
  // CHECK-NEXT:   0 | void D::z4()

  // CHECK-LABEL: VFTable for 'B' in 'F' (1 entry).
  // CHECK-NEXT:   0 | F::~F() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'F::~F()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'F' (1 entry).
  // CHECK-NEXT:   -- accessible via vfptr at offset 4 --
  // CHECK-NEXT:   0 | F::~F() [scalar deleting]
};

void build_vftable(F *obj) { delete obj; }

struct G : F {
  // CHECK-LABEL: VFTable for 'D' in 'F' in 'G' (1 entry).
  // CHECK-NEXT:   0 | void D::z4()

  // CHECK-LABEL: VFTable for 'B' in 'F' in 'G' (1 entry).
  // CHECK-NEXT:   0 | G::~G() [scalar deleting]
  // CHECK-NEXT:       [this adjustment: -4 non-virtual]

  // CHECK-LABEL: Thunks for 'G::~G()' (1 entry).
  // CHECK-NEXT:   0 | [this adjustment: -4 non-virtual]

  // CHECK-LABEL: VFTable indices for 'G' (1 entry).
  // CHECK-NEXT:   -- accessible via vfptr at offset 4 --
  // CHECK-NEXT:   0 | G::~G() [scalar deleting]
  virtual ~G();
};

void build_vftable(G *obj) { delete obj; }
