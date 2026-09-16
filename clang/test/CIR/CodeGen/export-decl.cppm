// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

export module Foo;

export void exportedFunc() {}
// CIR-DAG:  cir.func no_inline dso_local @_ZW3Foo12exportedFuncv
// LLVM-DAG: define dso_local void @_ZW3Foo12exportedFuncv

export {
  void exportedFunc2() {}
  int exportedVar = 42;
}
// CIR-DAG:  cir.func no_inline dso_local @_ZW3Foo13exportedFunc2v
// LLVM-DAG: define dso_local void @_ZW3Foo13exportedFunc2v

// CIR-DAG:  cir.global external @_ZW3Foo11exportedVar = #cir.int<42> : !s32i
// LLVM-DAG: @_ZW3Foo11exportedVar = global i32 42

// Not exported, but still has mangling/linkage.
void internalFunc() {}
// CIR-DAG:  cir.func no_inline dso_local @_ZW3Foo12internalFuncv
// LLVM-DAG: define dso_local void @_ZW3Foo12internalFuncv
