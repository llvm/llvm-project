// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=BEFORE --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t2.cir 2>&1
// RUN: FileCheck --check-prefix=AFTER --input-file=%t2.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

extern int bar();
void foo(void) __attribute__((constructor));
void foo(void) {
  bar();
}

// BEFORE: cir.func @_Z3foov() global_ctor(65535)

void foo2(void) __attribute__((constructor(777)));
void foo2(void) {
  bar();
}

// BEFORE: cir.func @_Z4foo2v() global_ctor(777)

void foo3(void) __attribute__((destructor));
void foo3(void) {
  bar();
}

// BEFORE: cir.func @_Z4foo3v() global_dtor(65535)

void foo4(void) __attribute__((destructor(789)));
void foo4(void) {
  bar();
}

// BEFORE: cir.func @_Z4foo4v() global_dtor(789)

// AFTER: module @{{.*}} attributes {cir.global_ctors = [#cir.global_ctor<"_Z3foov", 65535>, #cir.global_ctor<"_Z4foo2v", 777>], cir.global_dtors = [#cir.global_dtor<"_Z4foo3v", 65535>, #cir.global_dtor<"_Z4foo4v", 789>]
// LLVM: @llvm.global_ctors = appending constant [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_Z3foov, ptr null }, { i32, ptr, ptr } { i32 777, ptr @_Z4foo2v, ptr null }]
// LLVM-NEXT: @llvm.global_dtors = appending constant [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_Z4foo3v, ptr null }, { i32, ptr, ptr } { i32 789, ptr @_Z4foo4v, ptr null }]