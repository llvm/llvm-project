// RUN: %clang_cc1 -fno-rtti %s -emit-llvm -o %t -triple=i386-pc-win32 -fdump-vtable-layouts 2>&1 | FileCheck --check-prefix=VFTABLES %s
// RUN: FileCheck --check-prefix=GLOBALS %s < %t
// RUN: FileCheck --check-prefix=CODEGEN %s < %t

namespace test1 {

// Some covariant types.
struct A { int a; };
struct B { int b; };
struct C : A, B { int c; };
struct D : C { int d; };
struct E : D { int e; };

// One base class and two overrides, all with covariant return types.
struct H     { virtual B *foo(); };
struct I : H { virtual C *foo(); };
struct J : I { virtual D *foo(); J(); };
struct K : J { virtual E *foo(); K(); };

J::J() {}

// VFTABLES-LABEL: VFTable for 'test1::H' in 'test1::I' in 'test1::J' (3 entries).
// VFTABLES-NEXT:   0 | test1::D *test1::J::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::B *'): 4 non-virtual]
// VFTABLES-NEXT:   1 | test1::D *test1::J::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::C *'): 0 non-virtual]
// VFTABLES-NEXT:   2 | test1::D *test1::J::foo()

// GLOBALS-LABEL: @"\01??_7J@test1@@6B@" = linkonce_odr unnamed_addr constant { [3 x i8*] }
// GLOBALS: @"\01?foo@J@test1@@QAEPAUB@2@XZ"
// GLOBALS: @"\01?foo@J@test1@@QAEPAUC@2@XZ"
// GLOBALS: @"\01?foo@J@test1@@UAEPAUD@2@XZ"

K::K() {}

// VFTABLES-LABEL: VFTable for 'test1::H' in 'test1::I' in 'test1::J' in 'test1::K' (4 entries).
// VFTABLES-NEXT:   0 | test1::E *test1::K::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::B *'): 4 non-virtual]
// VFTABLES-NEXT:   1 | test1::E *test1::K::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::C *'): 0 non-virtual]
// VFTABLES-NEXT:   2 | test1::E *test1::K::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::D *'): 0 non-virtual]
// VFTABLES-NEXT:   3 | test1::E *test1::K::foo()

// Only B to C requires adjustment, but we get 3 thunks in K's vftable, two of
// which are trivial.
// GLOBALS-LABEL: @"\01??_7K@test1@@6B@" = linkonce_odr unnamed_addr constant { [4 x i8*] }
// GLOBALS: @"\01?foo@K@test1@@QAEPAUB@2@XZ"
// GLOBALS: @"\01?foo@K@test1@@QAEPAUC@2@XZ"
// GLOBALS: @"\01?foo@K@test1@@QAEPAUD@2@XZ"
// GLOBALS: @"\01?foo@K@test1@@UAEPAUE@2@XZ"

//  This thunk has a return adjustment.
// CODEGEN-LABEL: define {{.*}} @"\01?foo@K@test1@@QAEPAUB@2@XZ"
// CODEGEN: call {{.*}} @"\01?foo@K@test1@@UAEPAUE@2@XZ"
// CODEGEN: icmp {{.*}}, null
// CODEGEN: getelementptr
// CODEGEN: ret

//  These two don't.
// CODEGEN-LABEL: define {{.*}} @"\01?foo@K@test1@@QAEPAUC@2@XZ"
// CODEGEN: call {{.*}} @"\01?foo@K@test1@@UAEPAUE@2@XZ"
// CODEGEN-NEXT: ret

// CODEGEN-LABEL: define {{.*}} @"\01?foo@K@test1@@QAEPAUD@2@XZ"
// CODEGEN: call {{.*}} @"\01?foo@K@test1@@UAEPAUE@2@XZ"
// CODEGEN-NEXT: ret

}

namespace test2 {

// Covariant types.  D* is not trivially convertible to C*.
struct A { int a; };
struct B { int b; };
struct C : B { int c; };
struct D : A, C { int d; };
struct E : D { int e; };

// J's foo will require an adjusting thunk, and K will require a trivial thunk.
struct H     { virtual B *foo(); };
struct I : H { virtual C *foo(); };
struct J : I { virtual D *foo(); J(); };
struct K : J { virtual E *foo(); K(); };

J::J() {}

// VFTABLES-LABEL: VFTable for 'test2::H' in 'test2::I' in 'test2::J' (2 entries).
// VFTABLES-NEXT:    0 | test2::D *test2::J::foo()
// VFTABLES-NEXT:         [return adjustment (to type 'struct test2::B *'): 4 non-virtual]
// VFTABLES-NEXT:    1 | test2::D *test2::J::foo()

// GLOBALS-LABEL: @"\01??_7J@test2@@6B@" = linkonce_odr unnamed_addr constant { [2 x i8*] }

K::K() {}

// VFTABLES-LABEL: VFTable for 'test2::H' in 'test2::I' in 'test2::J' in 'test2::K' (3 entries).
// VFTABLES-NEXT:    0 | test2::E *test2::K::foo()
// VFTABLES-NEXT:         [return adjustment (to type 'struct test2::B *'): 4 non-virtual]
// VFTABLES-NEXT:    1 | test2::E *test2::K::foo()
// VFTABLES-NEXT:         [return adjustment (to type 'struct test2::D *'): 0 non-virtual]
// VFTABLES-NEXT:    2 | test2::E *test2::K::foo()

// GLOBALS-LABEL: @"\01??_7K@test2@@6B@" = linkonce_odr unnamed_addr constant { [3 x i8*] }

}

namespace pr20479 {
struct A {
  virtual A *f();
};

struct B : virtual A {
  virtual B *f();
};

struct C : virtual A, B {
// VFTABLES-LABEL: VFTable for 'pr20479::A' in 'pr20479::B' in 'pr20479::C' (2 entries).
// VFTABLES-NEXT:   0 | pr20479::B *pr20479::B::f()
// VFTABLES-NEXT:       [return adjustment (to type 'struct pr20479::A *'): vbase #1, 0 non-virtual]
// VFTABLES-NEXT:   1 | pr20479::B *pr20479::B::f()
  C();
};

C::C() {}

// GLOBALS-LABEL: @"\01??_7C@pr20479@@6B@" = linkonce_odr unnamed_addr constant { [2 x i8*] }
// GLOBALS: @"\01?f@B@pr20479@@QAEPAUA@2@XZ"
// GLOBALS: @"\01?f@B@pr20479@@UAEPAU12@XZ"
}

namespace pr21073 {
struct A {
  virtual A *f();
};

struct B : virtual A {
  virtual B *f();
};

struct C : virtual A, virtual B {
// VFTABLES-LABEL: VFTable for 'pr21073::A' in 'pr21073::B' in 'pr21073::C' (2 entries).
// VFTABLES-NEXT:   0 | pr21073::B *pr21073::B::f()
// VFTABLES-NEXT:       [return adjustment (to type 'struct pr21073::A *'): vbase #1, 0 non-virtual]
// VFTABLES-NEXT:       [this adjustment: 8 non-virtual]
// VFTABLES-NEXT:   1 | pr21073::B *pr21073::B::f()
// VFTABLES-NEXT:       [return adjustment (to type 'struct pr21073::B *'): 0 non-virtual]
// VFTABLES-NEXT:       [this adjustment: 8 non-virtual]
  C();
};

C::C() {}

// GLOBALS-LABEL: @"\01??_7C@pr21073@@6B@" = linkonce_odr unnamed_addr constant { [2 x i8*] }
// GLOBALS: @"\01?f@B@pr21073@@WPPPPPPPI@AEPAUA@2@XZ"
// GLOBALS: @"\01?f@B@pr21073@@WPPPPPPPI@AEPAU12@XZ"
}

namespace pr21073_2 {
struct A { virtual A *foo(); };
struct B : virtual A {};
struct C : virtual A { virtual C *foo(); };
struct D : B, C { D(); };
D::D() {}

// VFTABLES-LABEL: VFTable for 'pr21073_2::A' in 'pr21073_2::C' in 'pr21073_2::D' (2 entries)
// VFTABLES-NEXT:   0 | pr21073_2::C *pr21073_2::C::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct pr21073_2::A *'): vbase #1, 0 non-virtual]
// VFTABLES-NEXT:   1 | pr21073_2::C *pr21073_2::C::foo()

// GLOBALS-LABEL: @"\01??_7D@pr21073_2@@6B@" = {{.*}} constant { [2 x i8*] }
// GLOBALS: @"\01?foo@C@pr21073_2@@QAEPAUA@2@XZ"
// GLOBALS: @"\01?foo@C@pr21073_2@@UAEPAU12@XZ"
}

namespace test3 {
struct A { virtual A *fn(); };
struct B : virtual A { virtual B *fn(); };
struct X : virtual B {};
struct Y : virtual B {};
struct C : X, Y {};
struct D : virtual B, virtual A, C {
  D *fn();
  D();
};
D::D() {}

// VFTABLES-LABEL: VFTable for 'test3::A' in 'test3::B' in 'test3::X' in 'test3::C' in 'test3::D' (3 entries).
// VFTABLES-NEXT:   0 | test3::D *test3::D::fn()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test3::A *'): vbase #1, 0 non-virtual]
// VFTABLES-NEXT:       [this adjustment: vtordisp at -4, 0 non-virtual]
// VFTABLES-NEXT:   1 | test3::D *test3::D::fn()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test3::B *'): vbase #2, 0 non-virtual]
// VFTABLES-NEXT:       [this adjustment: vtordisp at -4, 0 non-virtual]
// VFTABLES-NEXT:   2 | test3::D *test3::D::fn()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test3::D *'): 0 non-virtual]
// VFTABLES-NEXT:       [this adjustment: vtordisp at -4, 0 non-virtual]

// GLOBALS-LABEL: @"\01??_7D@test3@@6B@" = {{.*}} constant { [3 x i8*] }
// GLOBALS: @"\01?fn@D@test3@@$4PPPPPPPM@A@AEPAUA@2@XZ"
// GLOBALS: @"\01?fn@D@test3@@$4PPPPPPPM@A@AEPAUB@2@XZ"
// GLOBALS: @"\01?fn@D@test3@@$4PPPPPPPM@A@AEPAU12@XZ"
}
