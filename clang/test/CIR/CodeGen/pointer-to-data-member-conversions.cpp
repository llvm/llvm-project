// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Base { int x; };
struct Derived : Base { double y; };

// BaseToDerivedMemberPointer, single inheritance with Base at offset 0 in
// Derived.  The cast adjustment is zero in this case.
int Derived::*b2d_simple = &Derived::x;
// CIR-BEFORE: cir.global external @b2d_simple = #cir.data_member<offset = 0> : !cir.data_member<!s32i in !rec_Derived>
// CIR-AFTER: cir.global external @b2d_simple = #cir.int<0> : !s64i
// LLVM: @b2d_simple = global i64 0
// OGCG: @b2d_simple = global i64 0

struct A { double a; };
struct B { int b; };
struct M : A, B { };

// BaseToDerivedMemberPointer with a non-zero path adjustment.  B is at
// offset 8 in M (after A's double), and B::b is at offset 0 in B, so the
// resolved offset of B::b viewed as a member of M is 8.
int M::*b2d_multi = &M::b;
// CIR-BEFORE: cir.global external @b2d_multi = #cir.data_member<offset = 8> : !cir.data_member<!s32i in !rec_M>
// CIR-AFTER: cir.global external @b2d_multi = #cir.int<8> : !s64i
// LLVM: @b2d_multi = global i64 8
// OGCG: @b2d_multi = global i64 8

// DerivedToBaseMemberPointer subtracts the adjustment, giving B::b at
// offset 0 in B.
int B::*d2b = (int B::*)&M::b;
// CIR-BEFORE: cir.global external @d2b = #cir.data_member<offset = 0> : !cir.data_member<!s32i in !rec_B>
// CIR-AFTER: cir.global external @d2b = #cir.int<0> : !s64i
// LLVM: @d2b = global i64 0
// OGCG: @d2b = global i64 0

struct Foo { int x; int y; };
struct Bar { float a; int b; };

// ReinterpretMemberPointer doesn't change the representation under
// Itanium -- Bar::b is at offset 4 in Bar, and the reinterpret as int
// Foo::* keeps the offset at 4.
int Foo::*reint = reinterpret_cast<int Foo::*>(&Bar::b);
// CIR-BEFORE: cir.global external @reint = #cir.data_member<offset = 4> : !cir.data_member<!s32i in !rec_Foo>
// CIR-AFTER: cir.global external @reint = #cir.int<4> : !s64i
// LLVM: @reint = global i64 4
// OGCG: @reint = global i64 4

// Null preserves null through all three cast kinds.
int Derived::*b2d_null = (int Derived::*)(int Base::*)nullptr;
// CIR-BEFORE: cir.global external @b2d_null = #cir.data_member<null> : !cir.data_member<!s32i in !rec_Derived>
// CIR-AFTER: cir.global external @b2d_null = #cir.int<-1> : !s64i
// LLVM: @b2d_null = global i64 -1
// OGCG: @b2d_null = global i64 -1

int B::*d2b_null = (int B::*)(int M::*)nullptr;
// CIR-BEFORE: cir.global external @d2b_null = #cir.data_member<null> : !cir.data_member<!s32i in !rec_B>
// CIR-AFTER: cir.global external @d2b_null = #cir.int<-1> : !s64i
// LLVM: @d2b_null = global i64 -1
// OGCG: @d2b_null = global i64 -1

int Foo::*reint_null = reinterpret_cast<int Foo::*>((int Bar::*)nullptr);
// CIR-BEFORE: cir.global external @reint_null = #cir.data_member<null> : !cir.data_member<!s32i in !rec_Foo>
// CIR-AFTER: cir.global external @reint_null = #cir.int<-1> : !s64i
// LLVM: @reint_null = global i64 -1
// OGCG: @reint_null = global i64 -1
