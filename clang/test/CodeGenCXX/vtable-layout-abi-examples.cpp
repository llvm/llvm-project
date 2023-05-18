// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm-only -fdump-vtable-layouts > %t 2>/dev/null
// RUN: FileCheck --check-prefix=CHECK-1 %s < %t
// RUN: FileCheck --check-prefix=CHECK-2 %s < %t
// RUN: FileCheck --check-prefix=CHECK-3 %s < %t
// RUN: FileCheck --check-prefix=CHECK-4 %s < %t
// RUN: FileCheck --check-prefix=CHECK-5 %s < %t
// RUN: FileCheck --check-prefix=CHECK-6 %s < %t
// RUN: FileCheck --check-prefix=CHECK-7 %s < %t
// RUN: FileCheck --check-prefix=CHECK-8 %s < %t
// RUN: FileCheck --check-prefix=CHECK-9 %s < %t
// RUN: FileCheck --check-prefix=CHECK-10 %s < %t
// RUN: FileCheck --check-prefix=CHECK-11 %s < %t

/// Examples from the Itanium C++ ABI specification.
/// http://www.codesourcery.com/public/cxx-abi/

namespace Test1 {
  
// This is from http://www.codesourcery.com/public/cxx-abi/cxx-vtable-ex.html

// CHECK-1:      Vtable for 'Test1::A' (5 entries).
// CHECK-1-NEXT:    0 | offset_to_top (0)
// CHECK-1-NEXT:    1 | Test1::A RTTI
// CHECK-1-NEXT:        -- (Test1::A, 0) vtable address --
// CHECK-1-NEXT:    2 | void Test1::A::f()
// CHECK-1-NEXT:    3 | void Test1::A::g()
// CHECK-1-NEXT:    4 | void Test1::A::h()
struct A {
  virtual void f ();
  virtual void g ();
  virtual void h ();
  int ia;
};
void A::f() {}

// CHECK-2:      Vtable for 'Test1::B' (13 entries).
// CHECK-2-NEXT:    0 | vbase_offset (16)
// CHECK-2-NEXT:    1 | offset_to_top (0)
// CHECK-2-NEXT:    2 | Test1::B RTTI
// CHECK-2-NEXT:        -- (Test1::B, 0) vtable address --
// CHECK-2-NEXT:    3 | void Test1::B::f()
// CHECK-2-NEXT:    4 | void Test1::B::h()
// CHECK-2-NEXT:    5 | vcall_offset (-16)
// CHECK-2-NEXT:    6 | vcall_offset (0)
// CHECK-2-NEXT:    7 | vcall_offset (-16)
// CHECK-2-NEXT:    8 | offset_to_top (-16)
// CHECK-2-NEXT:    9 | Test1::B RTTI
// CHECK-2-NEXT:        -- (Test1::A, 16) vtable address --
// CHECK-2-NEXT:   10 | void Test1::B::f()
// CHECK-2-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-2-NEXT:   11 | void Test1::A::g()
// CHECK-2-NEXT:   12 | void Test1::B::h()
// CHECK-2-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct B: public virtual A {
  void f ();
  void h ();
  int ib;
};
void B::f() {}

// CHECK-3:      Vtable for 'Test1::C' (13 entries).
// CHECK-3-NEXT:    0 | vbase_offset (16)
// CHECK-3-NEXT:    1 | offset_to_top (0)
// CHECK-3-NEXT:    2 | Test1::C RTTI
// CHECK-3-NEXT:        -- (Test1::C, 0) vtable address --
// CHECK-3-NEXT:    3 | void Test1::C::g()
// CHECK-3-NEXT:    4 | void Test1::C::h()
// CHECK-3-NEXT:    5 | vcall_offset (-16)
// CHECK-3-NEXT:    6 | vcall_offset (-16)
// CHECK-3-NEXT:    7 | vcall_offset (0)
// CHECK-3-NEXT:    8 | offset_to_top (-16)
// CHECK-3-NEXT:    9 | Test1::C RTTI
// CHECK-3-NEXT:        -- (Test1::A, 16) vtable address --
// CHECK-3-NEXT:   10 | void Test1::A::f()
// CHECK-3-NEXT:   11 | void Test1::C::g()
// CHECK-3-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-3-NEXT:   12 | void Test1::C::h()
// CHECK-3-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct C: public virtual A {
  void g ();
  void h ();
  int ic;
};
void C::g() {}

// CHECK-4:      Vtable for 'Test1::D' (18 entries).
// CHECK-4-NEXT:    0 | vbase_offset (32)
// CHECK-4-NEXT:    1 | offset_to_top (0)
// CHECK-4-NEXT:    2 | Test1::D RTTI
// CHECK-4-NEXT:        -- (Test1::B, 0) vtable address --
// CHECK-4-NEXT:        -- (Test1::D, 0) vtable address --
// CHECK-4-NEXT:    3 | void Test1::B::f()
// CHECK-4-NEXT:    4 | void Test1::D::h()
// CHECK-4-NEXT:    5 | vbase_offset (16)
// CHECK-4-NEXT:    6 | offset_to_top (-16)
// CHECK-4-NEXT:    7 | Test1::D RTTI
// CHECK-4-NEXT:        -- (Test1::C, 16) vtable address --
// CHECK-4-NEXT:    8 | void Test1::C::g()
// CHECK-4-NEXT:    9 | void Test1::D::h()
// CHECK-4-NEXT:        [this adjustment: -16 non-virtual]
// CHECK-4-NEXT:   10 | vcall_offset (-32)
// CHECK-4-NEXT:   11 | vcall_offset (-16)
// CHECK-4-NEXT:   12 | vcall_offset (-32)
// CHECK-4-NEXT:   13 | offset_to_top (-32)
// CHECK-4-NEXT:   14 | Test1::D RTTI
// CHECK-4-NEXT:        -- (Test1::A, 32) vtable address --
// CHECK-4-NEXT:   15 | void Test1::B::f()
// CHECK-4-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-4-NEXT:   16 | void Test1::C::g()
// CHECK-4-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-4-NEXT:   17 | void Test1::D::h()
// CHECK-4-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct D: public B, public C {
  void h ();
  int id;
};
void D::h() { }

struct X {
  int ix;
  virtual void x();
};

// CHECK-5:      Vtable for 'Test1::E' (24 entries).
// CHECK-5-NEXT:    0 | vbase_offset (56)
// CHECK-5-NEXT:    1 | offset_to_top (0)
// CHECK-5-NEXT:    2 | Test1::E RTTI
// CHECK-5-NEXT:        -- (Test1::E, 0) vtable address --
// CHECK-5-NEXT:        -- (Test1::X, 0) vtable address --
// CHECK-5-NEXT:    3 | void Test1::X::x()
// CHECK-5-NEXT:    4 | void Test1::E::f()
// CHECK-5-NEXT:    5 | void Test1::E::h()
// CHECK-5-NEXT:    6 | vbase_offset (40)
// CHECK-5-NEXT:    7 | offset_to_top (-16)
// CHECK-5-NEXT:    8 | Test1::E RTTI
// CHECK-5-NEXT:        -- (Test1::B, 16) vtable address --
// CHECK-5-NEXT:        -- (Test1::D, 16) vtable address --
// CHECK-5-NEXT:    9 | void Test1::E::f()
// CHECK-5-NEXT:        [this adjustment: -16 non-virtual]
// CHECK-5-NEXT:   10 | void Test1::E::h()
// CHECK-5-NEXT:        [this adjustment: -16 non-virtual]
// CHECK-5-NEXT:   11 | vbase_offset (24)
// CHECK-5-NEXT:   12 | offset_to_top (-32)
// CHECK-5-NEXT:   13 | Test1::E RTTI
// CHECK-5-NEXT:        -- (Test1::C, 32) vtable address --
// CHECK-5-NEXT:   14 | void Test1::C::g()
// CHECK-5-NEXT:   15 | void Test1::E::h()
// CHECK-5-NEXT:        [this adjustment: -32 non-virtual]
// CHECK-5-NEXT:   16 | vcall_offset (-56)
// CHECK-5-NEXT:   17 | vcall_offset (-24)
// CHECK-5-NEXT:   18 | vcall_offset (-56)
// CHECK-5-NEXT:   19 | offset_to_top (-56)
// CHECK-5-NEXT:   20 | Test1::E RTTI
// CHECK-5-NEXT:        -- (Test1::A, 56) vtable address --
// CHECK-5-NEXT:   21 | void Test1::E::f()
// CHECK-5-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-5-NEXT:   22 | void Test1::C::g()
// CHECK-5-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]
// CHECK-5-NEXT:   23 | void Test1::E::h()
// CHECK-5-NEXT:        [this adjustment: 0 non-virtual, -40 vcall offset offset]
struct E : X, D {
  int ie;
  void f();
  void h ();
};
void E::f() { } 

}

namespace Test2 {

// From http://www.codesourcery.com/public/cxx-abi/abi.html#class-types.

struct A { virtual void f(); };
struct B : virtual public A { int i; };
struct C : virtual public A { int j; };

// CHECK-6:      Vtable for 'Test2::D' (11 entries).
// CHECK-6-NEXT:    0 | vbase_offset (0)
// CHECK-6-NEXT:    1 | vcall_offset (0)
// CHECK-6-NEXT:    2 | offset_to_top (0)
// CHECK-6-NEXT:    3 | Test2::D RTTI
// CHECK-6-NEXT:        -- (Test2::A, 0) vtable address --
// CHECK-6-NEXT:        -- (Test2::B, 0) vtable address --
// CHECK-6-NEXT:        -- (Test2::D, 0) vtable address --
// CHECK-6-NEXT:    4 | void Test2::A::f()
// CHECK-6-NEXT:    5 | void Test2::D::d()
// CHECK-6-NEXT:    6 | vbase_offset (-16)
// CHECK-6-NEXT:    7 | vcall_offset (-16)
// CHECK-6-NEXT:    8 | offset_to_top (-16)
// CHECK-6-NEXT:    9 | Test2::D RTTI
// CHECK-6-NEXT:        -- (Test2::C, 16) vtable address --
// CHECK-6-NEXT:   10 | [unused] void Test2::A::f()
struct D : public B, public C {
  virtual void d();
};
void D::d() { } 

}

namespace Test3 {

// From http://www.codesourcery.com/public/cxx-abi/abi-examples.html#vtable-ctor

struct V1 {
  int v1;
  virtual void f();
};

struct V2 : virtual V1 {
  int v2;
  virtual void f();
};

// CHECK-7:      Vtable for 'Test3::C' (14 entries).
// CHECK-7-NEXT:    0 | vbase_offset (32)
// CHECK-7-NEXT:    1 | vbase_offset (16)
// CHECK-7-NEXT:    2 | offset_to_top (0)
// CHECK-7-NEXT:    3 | Test3::C RTTI
// CHECK-7-NEXT:        -- (Test3::C, 0) vtable address --
// CHECK-7-NEXT:    4 | void Test3::C::f()
// CHECK-7-NEXT:    5 | vcall_offset (-16)
// CHECK-7-NEXT:    6 | offset_to_top (-16)
// CHECK-7-NEXT:    7 | Test3::C RTTI
// CHECK-7-NEXT:        -- (Test3::V1, 16) vtable address --
// CHECK-7-NEXT:    8 | void Test3::C::f()
// CHECK-7-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-7-NEXT:    9 | vcall_offset (-32)
// CHECK-7-NEXT:   10 | vbase_offset (-16)
// CHECK-7-NEXT:   11 | offset_to_top (-32)
// CHECK-7-NEXT:   12 | Test3::C RTTI
// CHECK-7-NEXT:        -- (Test3::V2, 32) vtable address --
// CHECK-7-NEXT:   13 | void Test3::C::f()
// CHECK-7-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]

// CHECK-8:      Construction vtable for ('Test3::V2', 32) in 'Test3::C' (9 entries).
// CHECK-8-NEXT:    0 | vcall_offset (0)
// CHECK-8-NEXT:    1 | vbase_offset (-16)
// CHECK-8-NEXT:    2 | offset_to_top (0)
// CHECK-8-NEXT:    3 | Test3::V2 RTTI
// CHECK-8-NEXT:        -- (Test3::V2, 32) vtable address --
// CHECK-8-NEXT:    4 | void Test3::V2::f()
// CHECK-8-NEXT:    5 | vcall_offset (16)
// CHECK-8-NEXT:    6 | offset_to_top (16)
// CHECK-8-NEXT:    7 | Test3::V2 RTTI
// CHECK-8-NEXT:        -- (Test3::V1, 16) vtable address --
// CHECK-8-NEXT:    8 | void Test3::V2::f()
// CHECK-8-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
struct C : virtual V1, virtual V2 {
  int c;
  virtual void f();
};
void C::f() { }

struct B {
  int b;
};

// CHECK-9:      Vtable for 'Test3::D' (15 entries).
// CHECK-9-NEXT:    0 | vbase_offset (40)
// CHECK-9-NEXT:    1 | vbase_offset (24)
// CHECK-9-NEXT:    2 | offset_to_top (0)
// CHECK-9-NEXT:    3 | Test3::D RTTI
// CHECK-9-NEXT:        -- (Test3::C, 0) vtable address --
// CHECK-9-NEXT:        -- (Test3::D, 0) vtable address --
// CHECK-9-NEXT:    4 | void Test3::C::f()
// CHECK-9-NEXT:    5 | void Test3::D::g()
// CHECK-9-NEXT:    6 | vcall_offset (-24)
// CHECK-9-NEXT:    7 | offset_to_top (-24)
// CHECK-9-NEXT:    8 | Test3::D RTTI
// CHECK-9-NEXT:        -- (Test3::V1, 24) vtable address --
// CHECK-9-NEXT:    9 | void Test3::C::f()
// CHECK-9-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-9-NEXT:   10 | vcall_offset (-40)
// CHECK-9-NEXT:   11 | vbase_offset (-16)
// CHECK-9-NEXT:   12 | offset_to_top (-40)
// CHECK-9-NEXT:   13 | Test3::D RTTI
// CHECK-9-NEXT:        -- (Test3::V2, 40) vtable address --
// CHECK-9-NEXT:   14 | void Test3::C::f()
// CHECK-9-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]

// CHECK-10:      Construction vtable for ('Test3::C', 0) in 'Test3::D' (14 entries).
// CHECK-10-NEXT:    0 | vbase_offset (40)
// CHECK-10-NEXT:    1 | vbase_offset (24)
// CHECK-10-NEXT:    2 | offset_to_top (0)
// CHECK-10-NEXT:    3 | Test3::C RTTI
// CHECK-10-NEXT:        -- (Test3::C, 0) vtable address --
// CHECK-10-NEXT:    4 | void Test3::C::f()
// CHECK-10-NEXT:    5 | vcall_offset (-24)
// CHECK-10-NEXT:    6 | offset_to_top (-24)
// CHECK-10-NEXT:    7 | Test3::C RTTI
// CHECK-10-NEXT:        -- (Test3::V1, 24) vtable address --
// CHECK-10-NEXT:    8 | void Test3::C::f()
// CHECK-10-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
// CHECK-10-NEXT:    9 | vcall_offset (-40)
// CHECK-10-NEXT:   10 | vbase_offset (-16)
// CHECK-10-NEXT:   11 | offset_to_top (-40)
// CHECK-10-NEXT:   12 | Test3::C RTTI
// CHECK-10-NEXT:        -- (Test3::V2, 40) vtable address --
// CHECK-10-NEXT:   13 | void Test3::C::f()
// CHECK-10-NEXT:        [this adjustment: 0 non-virtual, -32 vcall offset offset]

// CHECK-11:      Construction vtable for ('Test3::V2', 40) in 'Test3::D' (9 entries).
// CHECK-11-NEXT:    0 | vcall_offset (0)
// CHECK-11-NEXT:    1 | vbase_offset (-16)
// CHECK-11-NEXT:    2 | offset_to_top (0)
// CHECK-11-NEXT:    3 | Test3::V2 RTTI
// CHECK-11-NEXT:        -- (Test3::V2, 40) vtable address --
// CHECK-11-NEXT:    4 | void Test3::V2::f()
// CHECK-11-NEXT:    5 | vcall_offset (16)
// CHECK-11-NEXT:    6 | offset_to_top (16)
// CHECK-11-NEXT:    7 | Test3::V2 RTTI
// CHECK-11-NEXT:        -- (Test3::V1, 24) vtable address --
// CHECK-11-NEXT:    8 | void Test3::V2::f()
// CHECK-11-NEXT:        [this adjustment: 0 non-virtual, -24 vcall offset offset]
struct D : B, C {
  int d;
  virtual void g();
};
void D::g() { }

}
