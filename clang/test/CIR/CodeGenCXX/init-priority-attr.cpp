// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM,OGCG

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM,OGCG

void foo(int);

class A {
public:
  A() { foo(1); }
};

class A1 {
public:
  A1() { foo(2); }
};

class B {
public:
  B() { foo(3); }
};

class C {
public:
  static A a;
  C() { foo(4); }
};


A C::a = A();


// Static inside of C.
// CIR-BEFORE-LPP: cir.global external @_ZN1C1aE = ctor : !rec_A {
// CIR-BEFORE-LPP: } {alignment = 1 : i64, ast = #cir.var.decl.ast}
// CIR-BEFORE-LPP: cir.global external @c = ctor : !rec_C {
// CIR-BEFORE-LPP: } {alignment = 1 : i64, ast = #cir.var.decl.ast}

// CIR-BEFORE-LPP: cir.global external @a1 = ctor : !rec_A1 {
// CIR-BEFORE-LPP: } {alignment = 1 : i64, ast = #cir.var.decl.ast, init_priority = 300 : i32}
// CIR-BEFORE-LPP: cir.global external @a = ctor : !rec_A {
// CIR-BEFORE-LPP: } {alignment = 1 : i64, ast = #cir.var.decl.ast, init_priority = 300 : i32}
// CIR-BEFORE-LPP: cir.global external @b = ctor : !rec_B {
// CIR-BEFORE-LPP: } {alignment = 1 : i64, ast = #cir.var.decl.ast, init_priority = 200 : i32}

// CIR: cir.global_ctors = [#cir.global_ctor<"_GLOBAL__I_000200", 200>, #cir.global_ctor<"_GLOBAL__I_000300", 300>, #cir.global_ctor<"_GLOBAL__sub_I_[[FILENAME:.*]]", 65535>]
// CIR-LABEL: cir.func internal private @__cxx_global_var_init() {
// CIR-NEXT:    cir.get_global @_ZN1C1aE : !cir.ptr<!rec_A>
// CIR-NEXT:    cir.call @_ZN1AC1Ev(
// CIR-NEXT:    cir.return
//
// CIR-LABEL: cir.func internal private @__cxx_global_var_init.1() {
// CIR-NEXT:    cir.get_global @c : !cir.ptr<!rec_C>
// CIR-NEXT:    cir.call @_ZN1CC1Ev(
// CIR-NEXT:    cir.return

// CIR-LABEL: cir.func internal private @__cxx_global_var_init.2() {
// CIR-NEXT:    cir.get_global @a1 : !cir.ptr<!rec_A1> loc(#loc36)
// CIR-NEXT:    cir.call @_ZN2A1C1Ev(
// CIR-NEXT:    cir.return

// CIR-LABEL: cir.func internal private @__cxx_global_var_init.4() {
// CIR-NEXT:    cir.get_global @b : !cir.ptr<!rec_B> loc(#loc38)
// CIR-NEXT:    cir.call @_ZN1BC1Ev(
// CIR-NEXT:    cir.return

// CIR-LABEL: cir.func internal private @_GLOBAL__I_000200() {
// CIR-NEXT:   cir.call @__cxx_global_var_init.4() : () -> ()
// CIR-NEXT:   cir.return

// CIR-LABEL: cir.func internal private @_GLOBAL__I_000300() {
// CIR-NEXT:   cir.call @__cxx_global_var_init.2() : () -> ()
// CIR-NEXT:   cir.call @__cxx_global_var_init.3() : () -> ()
// CIR-NEXT:   cir.return

// CIR-LABEL: cir.func internal private @_GLOBAL__sub_I_
// CIR-SAME: [[FILENAME]]() {
// CIR-NEXT:   cir.call @__cxx_global_var_init() : () -> ()
// CIR-NEXT:   cir.call @__cxx_global_var_init.1() : () -> ()
// CIR-NEXT:   cir.return

// LLVM: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 200, ptr @_GLOBAL__I_000200, ptr null }, { i32, ptr, ptr } { i32 300, ptr @_GLOBAL__I_000300, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_[[FILENAME:.*]], ptr null }]

// LLVM-LABEL: define internal void @__cxx_global_var_init()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @_ZN1AC1Ev({{.*}}@_ZN1C1aE)

// LLVM-LABEL: define internal void @__cxx_global_var_init.1()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @_ZN1CC1Ev({{.*}}@c)

// LLVM-LABEL: define internal void @__cxx_global_var_init.2()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @_ZN2A1C1Ev({{.*}}@a1)

// LLVM-LABEL: define internal void @__cxx_global_var_init.3()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @_ZN1AC1Ev({{.*}}@a)

// LLVM-LABEL: define internal void @__cxx_global_var_init.4()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @_ZN1BC1Ev({{.*}}@b)

// LLVM-LABEL: define internal void @_GLOBAL__I_000200()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @__cxx_global_var_init.4()
// LLVM-NEXT:   ret void

// LLVM-LABEL: define internal void @_GLOBAL__I_000300()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @__cxx_global_var_init.2()
// LLVM-NEXT:   call void @__cxx_global_var_init.3()
// LLVM-NEXT:   ret void

// LLVM-LABEL: define internal void @_GLOBAL__sub_I_
// LLVM-SAME: [[FILENAME]]()
// OGCG-NEXT: entry:
// LLVM-NEXT:   call void @__cxx_global_var_init()
// LLVM-NEXT:   call void @__cxx_global_var_init.1()
// LLVM-NEXT:   ret void


C c;
A1 a1 __attribute__((init_priority (300)));
A a __attribute__((init_priority (300)));
B b __attribute__((init_priority (200)));
