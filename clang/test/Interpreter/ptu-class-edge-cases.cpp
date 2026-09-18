// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl | FileCheck %s
//
// Class-specific PTU rollback/recovery regression tests. Each scenario is
// adapted from a real clang/test/SemaCXX case and reshaped into: a
// committed baseline, a same-line chunk that mixes a legitimate new class
// with a hard Sema error (forcing IncrementalParser to roll the whole
// chunk back through PTUSlabRollback), then a follow-up chunk that
// re-exercises the class-specific machinery the rolled-back chunk touched.

extern "C" int printf(const char *, ...);

// 1. Virtual inheritance / diamond layout, cf. SemaCXX/long-virtual-inheritance-chain.cpp.
// Virtual bases force VTT and virtual-base-offset computation
// (ASTRecordLayouts, OverriddenMethods, KeyFunctions) beyond what a
// non-virtual hierarchy needs.
struct VDiamondBase { virtual int who() { return 0; } virtual ~VDiamondBase() {} };
struct VDiamondLeft : virtual VDiamondBase { int who() override { return 1; } };
struct VDiamondRight : virtual VDiamondBase { int who() override { return 2; } };
struct VDiamondJoin : VDiamondLeft, VDiamondRight { int who() override { return 3; } };
VDiamondJoin vdj;
auto rc1 = printf("vdj.who()=%d\n", vdj.who());
// CHECK: vdj.who()=3

struct VBad : virtual VDiamondBase { int who() override { return 9; } }; intentional_error_type vbad_garbage;
VDiamondJoin vdj2;
VDiamondBase* vbp = &vdj2;
auto rc2 = printf("vbp->who()=%d\n", vbp->who());
// CHECK: vbp->who()=3

// 2. Abstract class instantiation, cf. SemaCXX/abstract.cpp. Uses the real
// "allocating an object of abstract class type" diagnostic (rather than a
// synthetic bad token) as the rollback trigger, alongside a legitimate
// concrete subclass on the same line.
struct AbstractShape { virtual double area() const = 0; virtual ~AbstractShape() {} };
struct ConcreteSquare : AbstractShape { double side = 4.0; double area() const override { return side * side; } };
ConcreteSquare sq1;
auto rc3 = printf("sq1.area()=%.1f\n", sq1.area());
// CHECK: sq1.area()=16.0

struct ConcreteTriangle : AbstractShape { double base_len = 6.0, height = 2.0; double area() const override { return 0.5 * base_len * height; } }; AbstractShape *bad_instance = new AbstractShape();
struct ConcreteTriangle : AbstractShape { double base_len = 6.0, height = 2.0; double area() const override { return 0.5 * base_len * height; } };
ConcreteTriangle tri1;
auto rc4 = printf("tri1.area()=%.1f\n", tri1.area());
// CHECK: tri1.area()=6.0

// 3. Anonymous union member lookup, cf. SemaCXX/anonymous-union.cpp.
// Anonymous unions/structs synthesize IndirectFieldDecl chains that member
// lookup has to see through.
struct VariantBox { union { int as_int; float as_float; }; bool is_int; };
VariantBox vb1; vb1.is_int = true; vb1.as_int = 55;
auto rc5 = printf("vb1.as_int=%d\n", vb1.as_int);
// CHECK: vb1.as_int=55

struct VariantBoxBad { union { int as_int; struct { short lo; short hi; }; }; }; intentional_error_type variant_garbage;
VariantBox vb2; vb2.is_int = false; vb2.as_float = 2.5f;
auto rc6 = printf("vb2.as_float=%.1f is_int=%d\n", vb2.as_float, vb2.is_int);
// CHECK: vb2.as_float=2.5 is_int=0

// 4. Inheriting constructors / using-declarations, cf.
// SemaCXX/cxx11-inheriting-ctors.cpp and SemaCXX/using-decl.cpp.
// `using Base::Base;` synthesizes CXXConstructorDecls backed by
// UsingShadowDecl bookkeeping distinct from ordinary member lookup.
struct IctorBase { int val; IctorBase(int v) : val(v) {} };
struct IctorDerived : IctorBase { using IctorBase::IctorBase; };
IctorDerived id1(7);
auto rc7 = printf("id1.val=%d\n", id1.val);
// CHECK: id1.val=7

struct IctorDerivedBad : IctorBase { using IctorBase::IctorBase; int extra = 0; }; intentional_error_type ictor_garbage;
IctorDerived id2(13);
auto rc8 = printf("id2.val=%d\n", id2.val);
// CHECK: id2.val=13

// 5. Bit-field record layout, cf. SemaCXX/bitfield.cpp. Bit-fields exercise
// a layout code path (bit offsets within a storage unit) distinct from
// plain field layout, cached alongside ASTRecordLayouts.
struct Flags { unsigned a : 3; unsigned b : 5; unsigned c : 1; };
Flags fl1; fl1.a = 5; fl1.b = 20; fl1.c = 1;
auto rc9 = printf("fl1.a=%u fl1.b=%u fl1.c=%u sizeof=%lu\n", fl1.a, fl1.b, fl1.c, sizeof(Flags));
// CHECK: fl1.a=5 fl1.b=20 fl1.c=1

struct FlagsBad { unsigned x : 4; unsigned y : 30; }; intentional_error_type bitfield_garbage;
Flags fl2; fl2.a = 7; fl2.b = 31; fl2.c = 0;
auto rc10 = printf("fl2.a=%u fl2.b=%u fl2.c=%u sizeof=%lu\n", fl2.a, fl2.b, fl2.c, sizeof(Flags));
// CHECK: fl2.a=7 fl2.b=31 fl2.c=0

// 6. Pointer-to-member-function targeting a virtual method, cf.
// SemaCXX/member-pointer.cpp. Dereferencing such a pointer needs the
// vtable index of the target, tying MemberPointerType caching to the same
// virtual-dispatch bookkeeping stressed in test 1.
struct MPBase { virtual int compute(int x) { return x + 1; } virtual ~MPBase() {} };
struct MPDerived : MPBase { int compute(int x) override { return x * 2; } };
int (MPBase::*mp1)(int) = &MPBase::compute;
MPDerived mpd1;
auto rc11 = printf("(mpd1.*mp1)(5)=%d\n", (mpd1.*mp1)(5));
// CHECK: (mpd1.*mp1)(5)=10

struct MPDerivedBad : MPBase { int compute(int x) override { return x * 3; } }; int (MPBase::*mp_bad)(int) = &MPBase::compute; intentional_error_type mp_garbage;
MPDerived mpd2;
int (MPBase::*mp2)(int) = &MPBase::compute;
auto rc12 = printf("(mpd2.*mp2)(5)=%d\n", (mpd2.*mp2)(5));
// CHECK: (mpd2.*mp2)(5)=10

%quit
