// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Base { int x; };
struct Derived : Base { double y; };

int Derived::*b2d_simple = &Derived::x;
// CIR-BEFORE: cir.global external @b2d_simple = ctor
// CIR-BEFORE: cir.const #cir.data_member<0> : !cir.data_member<!s32i in !rec_Base>
// CIR-BEFORE: cir.derived_data_member %{{.*}}[0] : !cir.data_member<!s32i in !rec_Base> -> !cir.data_member<!s32i in !rec_Derived>
// CIR-AFTER: cir.func{{.*}}@__cxx_global_var_init
// CIR-AFTER: cir.const #cir.int<0> : !s64i
// LLVM-DAG: define internal void @__cxx_global_var_init()
// LLVM-DAG: store i64 0
// OGCG: @b2d_simple = global i64 0

struct A { double a; };
struct B { int b; };
struct M : A, B { };

int M::*b2d_multi = &M::b;
// CIR-BEFORE: cir.global external @b2d_multi = ctor
// CIR-BEFORE: cir.const #cir.data_member<0> : !cir.data_member<!s32i in !rec_B>
// CIR-BEFORE: cir.derived_data_member %{{.*}}[8] : !cir.data_member<!s32i in !rec_B> -> !cir.data_member<!s32i in !rec_M>
// CIR-AFTER: cir.func{{.*}}@__cxx_global_var_init.1
// CIR-AFTER: cir.const #cir.int<8> : !s64i
// LLVM-DAG: define internal void @__cxx_global_var_init.1()
// LLVM-DAG: store i64 8
// OGCG: @b2d_multi = global i64 8

int B::*d2b = (int B::*)&M::b;
// CIR-BEFORE: cir.global external @d2b = #cir.data_member<0> : !cir.data_member<!s32i in !rec_B>
// CIR-AFTER: cir.global external @d2b = #cir.int<0> : !s64i
// OGCG: @d2b = global i64 0

struct Foo { int x; int y; };
struct Bar { float a; int b; };

int Foo::*reint = reinterpret_cast<int Foo::*>(&Bar::b);
// CIR-BEFORE: cir.global external @reint = #cir.data_member<1> : !cir.data_member<!s32i in !rec_Foo>
// CIR-AFTER: cir.global external @reint = #cir.int<4> : !s64i
// OGCG: @reint = global i64 4

int Derived::*b2d_null = (int Derived::*)(int Base::*)nullptr;
// CIR-BEFORE: cir.global external @b2d_null = #cir.data_member<null> : !cir.data_member<!s32i in !rec_Derived>
// CIR-AFTER: cir.global external @b2d_null = #cir.int<-1> : !s64i
// LLVM-DAG: @b2d_null = global i64 -1
// OGCG: @b2d_null = global i64 -1

int B::*d2b_null = (int B::*)(int M::*)nullptr;
// CIR-BEFORE: cir.global external @d2b_null = #cir.data_member<null> : !cir.data_member<!s32i in !rec_B>
// CIR-AFTER: cir.global external @d2b_null = #cir.int<-1> : !s64i
// LLVM-DAG: @d2b_null = global i64 -1
// OGCG: @d2b_null = global i64 -1

int Foo::*reint_null = reinterpret_cast<int Foo::*>((int Bar::*)nullptr);
// CIR-BEFORE: cir.global external @reint_null = #cir.data_member<null> : !cir.data_member<!s32i in !rec_Foo>
// CIR-AFTER: cir.global external @reint_null = #cir.int<-1> : !s64i
// LLVM-DAG: @reint_null = global i64 -1
// OGCG: @reint_null = global i64 -1
