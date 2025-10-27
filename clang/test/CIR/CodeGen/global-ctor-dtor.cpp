// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR-AFTER
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

extern int bar();
void foo(void) __attribute__((constructor));
void foo(void) {
  bar();
}

// CIR-BEFORE-LPP: cir.func dso_local @_Z3foov() global_ctor

void foo2(void) __attribute__((constructor(777)));
void foo2(void) {
  bar();
}

// CIR-BEFORE-LPP: cir.func dso_local @_Z4foo2v() global_ctor(777)

void foo3(void) __attribute__((destructor));
void foo3(void) {
  bar();
}

// CIR-BEFORE-LPP: cir.func dso_local @_Z4foo3v() global_dtor

void foo4(void) __attribute__((destructor(789)));
void foo4(void) {
  bar();
}

// CIR-BEFORE-LPP: cir.func dso_local @_Z4foo4v() global_dtor(789)

// CIR-AFTER: module @{{.*}} attributes {cir.global_ctors = [#cir.global_ctor<"_Z3foov", 65535>, #cir.global_ctor<"_Z4foo2v", 777>], cir.global_dtors = [#cir.global_dtor<"_Z4foo3v", 65535>, #cir.global_dtor<"_Z4foo4v", 789>]

// LLVM: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_Z3foov, ptr null }, { i32, ptr, ptr } { i32 777, ptr @_Z4foo2v, ptr null }]
// LLVM: @llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_Z4foo3v, ptr null }, { i32, ptr, ptr } { i32 789, ptr @_Z4foo4v, ptr null }]

// OGCG: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_Z3foov, ptr null }, { i32, ptr, ptr } { i32 777, ptr @_Z4foo2v, ptr null }]
// OGCG: @llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_Z4foo3v, ptr null }, { i32, ptr, ptr } { i32 789, ptr @_Z4foo4v, ptr null }]
